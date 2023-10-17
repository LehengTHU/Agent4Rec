from simulation.base.abstract_memory import abstract_memory
import datetime
from langchain.schema import BaseMemory, Document
import openai
import logging
import re
from langchain.schema import BaseMemory, Document
from langchain.utils import mock_now
import time
import datetime
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from langchain.schema import BaseRetriever, Document
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
from langchain.chat_models import ChatOpenAI
from langchain.schema.language_model import BaseLanguageModel
from simulation.retriever import AvatarRetriver

import wandb

from termcolor import cprint

import simulation.vars as vars

class AvatarMemory(BaseMemory):
    llm: BaseLanguageModel
    """The core language model."""
    memory_retriever: AvatarRetriver
    """The retriever to fetch related memories."""
    reflection_threshold: Optional[float] = None
    """When aggregate_importance exceeds reflection_threshold, stop to reflect."""
    importance_weight: float = 0.15
    """How much weight to assign the memory importance."""
    aggregate_importance: float = 0.0  # : :meta private:
    """Track the sum of the 'importance' of recent memories.
    Triggers reflection when it reaches reflection_threshold."""
    reflecting: bool = False
    now_key: str = "now"
    max_tokens_limit: int = 1200  # : :meta private:

    user_k_tokens: float = 0.0
    use_wandb: bool = False


    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save the context of this model run to memory."""
        # TODO

    @property
    def memory_variables(self) -> List[str]:
        """Input keys this memory class will load dynamically."""
        # TODO

    def clear(self) -> None:
        """Clear memory contents."""
        # TODO

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return key-value pairs given the text input to the chain."""
        # TODO

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r"\n", text.strip())
        lines = [line for line in lines if line.strip()]  # remove empty lines
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def fetch_memories(
        self, observation: str, now: Optional[datetime.datetime] = None
    ) -> List[Document]:
        """Fetch related memories."""
        #print(observation)
        #print(now)
        return self.memory_retriever.get_relevant_documents(observation,now)

    def format_memories_detail(self, relevant_memories: List[Document]) -> str:
        content = []
        for mem in relevant_memories:
            content.append(self._format_memory_detail(mem, prefix="- "))
        return "\n".join([f"{mem}" for mem in content])

    def _format_memory_detail(self, memory: Document, prefix: str = "") -> str:
        created_time = memory.metadata["created_at"].strftime("%B %d, %Y, %I:%M %p")
        return f"{prefix}{memory.page_content.strip()}"

    def format_memories_simple(self, relevant_memories: List[Document]) -> str:
        return "; ".join([f"{mem.page_content}" for mem in relevant_memories])

    def _get_memories_until_limit(self, consumed_tokens: int) -> str:
        """Reduce the number of tokens in the documents."""
        result = []
        for doc in self.memory_retriever.memory_stream[::-1]:
            if consumed_tokens >= self.max_tokens_limit:
                break
            consumed_tokens += self.llm.get_num_tokens(doc.page_content)
            if consumed_tokens < self.max_tokens_limit:
                result.append(doc)
        return self.format_memories_simple(result)

    def get_completion(self, prompt, llm="gpt-3.5-turbo", temperature=0):
        messages = [{"role":"user", "content" : prompt}]
        response = ''
        except_waiting_time = 1
        total_waiting_time = 0
        max_waiting_time = 16
        current_sleep_time = 0.5
        while response == '':
            try:
                if(self.use_wandb): # whether to use wandb
                    start_time = time.time()

                    if((start_time - vars.global_start_time)//vars.global_interval > vars.global_steps):
                        if(vars.lock.acquire(False)):
                            print("??", vars.lock, (start_time - vars.global_start_time)//vars.global_interval, vars.global_interval, vars.global_steps)
                            print("\nMemory Start Identifier", start_time, vars.global_start_time, (start_time - vars.global_start_time), vars.global_steps)
                            # vars.lock = True
                            vars.global_steps += 1
                            wandb.log(
                                data = {"Real-time Traffic": vars.global_k_tokens - vars.global_last_tokens_record,
                                        "Total Traffic": vars.global_k_tokens,
                                        "Finished Users": vars.global_finished_users,
                                        "Finished Pages": vars.global_finished_pages,
                                        "Error Cast": vars.global_error_cast/1000
                                },
                                step = vars.global_steps
                            )
                            vars.global_last_tokens_record = vars.global_k_tokens
                            # vars.lock = False
                            vars.lock.release()
                            print("\nMemory End Identifier", time.time(), vars.global_start_time, (time.time() - vars.global_start_time), vars.global_steps)

                response = openai.ChatCompletion.create(
                    model=llm,
                    messages=messages,
                    temperature=temperature,
                    request_timeout = 20,
                    max_tokens=1000
                )

                print("===================================")
                print(f'{response["usage"]["total_tokens"]} = {response["usage"]["prompt_tokens"]} + {response["usage"]["completion_tokens"]} tokens counted by the OpenAI API.')
                k_tokens = response["usage"]["total_tokens"]/1000
                self.user_k_tokens += k_tokens
                vars.global_k_tokens += k_tokens
                if(response["usage"]["prompt_tokens"] > 2000):
                    cprint(prompt, color="white")
            
            except Exception as e:
                vars.global_error_cast += 1
                total_waiting_time += except_waiting_time
                time.sleep(current_sleep_time)
                if except_waiting_time < max_waiting_time:
                    except_waiting_time *= 2
                    current_sleep_time = np.random.randint(0, except_waiting_time-1)

        return response.choices[0].message["content"]
    
    def _user_taste_reflection(self, last_k: int = 10) -> List[str]:
        """Return the user's taste about recent movies."""
        prompt = """
            The user has watched following movie recently:
            <INPUT>\n\n
            Given only the information above, conclude the user's taste of movie using five adjective words, which should be conclusive, descriptive and movie-genre related.
            The output format must be: 
            user's recent taste are: <word1>,<word2>,<word3>,<word4>,<word5>.
            """
        
        observations = self.memory_retriever.memory_stream[-last_k:]
        observation_str = "\n".join(
            [self._format_memory_detail(o) for o in observations]
        )
        prompt_filled = prompt.replace("<INPUT>", observation_str)
        result = self.get_completion(prompt=prompt_filled, llm="gpt-3.5-turbo", temperature=0.2)
        print(result)
        return result
    
    def _user_satisfaction_reflection(self, last_k: int = 10) -> List[str]:
        """Return the user's feeling about recent movies."""
        prompt = """
            <INPUT>\n\n
            Given only the information above, describe your feeling of the recommendation result using a sentence. 
            The output format must be:
            [unsatisfied/satisfied] with the recommendation result because [reason].
            """
        
        
        observations = "what's your interaction history with each page of recommender?"
        relevant_memories = self.fetch_memories(observations)
        observation_str = self.format_memories_detail(relevant_memories)
        prompt_filled = prompt.replace("<INPUT>", observation_str)
        result = self.get_completion(prompt=prompt_filled, llm="gpt-3.5-turbo", temperature=0.2)

        print(result)
        return result
        # return "satisfaction reflected"
    
    def _user_feeling_reflection(self, last_k: int = 10) -> List[str]:
        """Return the user's feeling about recent movies."""
        #user persona: <INPUT 1>
        prompt = """
            user persona: a 22-year-old woman working in a clerical/administrative role. She is intelligent, imaginative, and adventurous. With a passion for movies, Emily has a diverse taste and enjoys a wide range of genres. Her favorite films include ""Princess Bride,"" ""Fried Green Tomatoes,"" ""Men in Black,"" ""Cinderella,"" ""Elizabeth,"" ""Star Wars: Episode V - The Empire Strikes Back,"" ""Ghost in the Shell,"" ""Mad Max 2,"" ""Usual Suspects,"" ""My Left Foot,"" ""Last Emperor,"" ""Dangerous Liaisons,"" ""Misérables,"" ""Howards End,"" and ""Spy Who Loved Me."" Emily's movie preferences reflect her love for captivating stories, fantasy, action, and historical dramas. She appreciates thought-provoking narratives and enjoys exploring different worlds through cinema."
            3,3,"Sound of Music, The (1965); Star Wars: Episode IV - A New Hope (1977); Fish Called Wanda, A (1988); One Flew Over the Cuckoo's Nest (1975); Silence of the Lambs, The (1991); Dead Poets Society (1989); Goldfinger (1964); To Kill a Mockingbird (1962); Reservoir Dogs (1992); Witness (1985); Steel Magnolias (1989); Godfather: Part II, The (1974); In the Line of Fire (1993); Shawshank Redemption, The (1994); Seven (Se7en) (1995)",Musical; Action|Adventure|Fantasy|Sci-Fi; Comedy; Drama; Drama|Thriller; Drama; Action; Drama; Crime|Thriller; Drama|Romance|Thriller; Drama; Action|Crime|Drama; Action|Thriller; Drama; Crime|Thriller,Male,45-49,clerical/admin,55421,"<Part 1>

            This user has watched following movies recently:
            <INPUT 2>\n\n
            Given only the information above, describe the user's feeling of each of the movie he/she watched recently.
            """
        
        observations = self.memory_retriever.memory_stream[-last_k:]
        observation_str = "\n".join(
            [self._format_memory_detail(o) for o in observations]
        )
        prompt_filled = prompt.replace("<INPUT 2>", observation_str)
        result = self.get_completion(prompt=prompt_filled, llm="gpt-3.5-turbo", temperature=0.2)
        print(result)
        return result

    def pause_to_reflect_taste(self, now: Optional[datetime.datetime] = None) -> List[str]:
        """Reflect on recent observations and generate 'insights'."""
        taste = self._user_taste_reflection()
        self.add_memory(taste, now=now)
        return 'taste reflected:\n'+ taste
    
    def pause_to_reflect_satisfaction(self, now: Optional[datetime.datetime] = None) -> List[str]:
        """Reflect on recent observations and generate 'insights'."""
        satisfaction = self._user_satisfaction_reflection()
        self.add_memory(satisfaction, now=now)
        return 'satisfaction reflected:\n'+ satisfaction
    
    def pause_to_reflect_feeling(self, now: Optional[datetime.datetime] = None) -> List[str]:
        """Reflect on recent observations and generate 'insights'."""
        feeling = self._user_feeling_reflection()
        self.add_memory(feeling, now=now)
        return 'feeling reflected:\n'+ feeling

    def add_memory(
        self, memory_content: str, now: Optional[datetime.datetime] = None
    ) -> List[str]:
        """Add an observation or memory to the agent's memory bank."""
        importance_score = 1
        self.aggregate_importance += importance_score
        document = Document(
            page_content=memory_content, metadata={"importance": importance_score}
        )
        result = self.memory_retriever.add_documents([document], current_time=now)

        # After an agent has processed a certain amount of memories (as measured by
        # aggregate importance), it is time to reflect on recent events to add
        # more synthesized memories to the agent's memory stream.
        if (
            self.reflection_threshold is not None
            and self.aggregate_importance > self.reflection_threshold
            and not self.reflecting
        ):
            self.reflecting = True
            self.reflect(now=now)
            # Hack to clear the importance from reflection
            self.aggregate_importance = 0.0
            self.reflecting = False
        return result
    
    def update_memory(self, reaction):
        """
        Update the memory bank with the reaction
        """
        return

    
    def time_weighting(self):
        """
        Weighting the memory according to the time
        """
        raise NotImplementedError
    
    def importance_weighting(self):
        """
        Weighting the importance of memory according to 
        the results of recommendation and the personal taste
        """
        raise NotImplementedError
    
    def reflect(self, now: Optional[datetime.datetime] = None):
        """
        Generate a high level understanding of previous memories
        """
        # self.pause_to_reflect_taste(now=now)
        # self.pause_to_reflect_feeling(now=now)
        self.pause_to_reflect_satisfaction(now=now)
        return 0