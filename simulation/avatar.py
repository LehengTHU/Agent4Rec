from simulation.base.abstract_avatar import abstract_avatar
from simulation.memory import AvatarMemory

from termcolor import colored, cprint
import openai
import os

import re
import numpy as np
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.chat_models import ChatOpenAI
from simulation.retriever import AvatarRetriver
import time
import datetime
import torch
from langchain.embeddings import OpenAIEmbeddings
import pandas as pd

import wandb

import simulation.vars as vars

class Avatar(abstract_avatar):
    def __init__(self, args, avatar_id, init_property, init_statistic):
        super().__init__(args, avatar_id)

        self.parse_init_property(init_property)
        self.parse_init_statistic(init_statistic)
        self.log_file = f"storage/{args.dataset}/{args.modeltype}/{args.simulation_name}/running_logs/{avatar_id}.txt"
        self.user_most_like = pd.read_pickle('/storage_fast/lhsheng/lhsheng/Sandbox/Simulator/scripts/user_genre_dict.pkl')
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        self.init_memory()

    def parse_init_property(self, init_property):
        """
        Parse the init property of the avatar
        """
        # self.taste = init_property["taste"].split("| ")
        self.taste = init_property["taste"].split("| ")
        # reason = init_property["reason"].split("| ")
        # self.taste = [taste[i] + " because " + reason[i] for i in range(len(taste))]
        self.high_rating = init_property["high_rating"]
        #self.low_rating = "should give low ratings to movies that don't have above genres or movies that have low historical ratings."
        # self.low_rating = init_property["low_rating"]
        #self.reason = init_property["reason"].split("| ")
        #self.movie = init_property["movie"].split("| ")
        #print(self.taste, self.reason, self.movie)

    def parse_init_statistic(self, init_statistic):
        """
        Parse the init statistic of the avatar
        """
# diversity_dict
        activity_dict = {   1:"An Incredibly Elusive Occasional Viewer, so seldom attracted by movie recommendations that it's almost a legendary event when you do watch a movie. Your movie-watching habits are extraordinarily infrequent. And you will exit the recommender system immediately even if you just feel little unsatisfied.",
                            #2:"A Casual Watcher who watches movies occasionally in the system. You only have moderate interest in recommendations. And you tend to exit recommendation system if you have some unsatisfied memory.",
                            2:"An Occasional Viewer, seldom attracted by movie recommendations. Only curious about watching movies that strictly align the taste. The movie-watching habits are not very infrequent. And you tend to exit the recommender system if you have a few unsatisfied memories.",
                            3:"A Movie Enthusiast with an insatiable appetite for films, willing to watch nearly every movie recommended to you. Movies are a central part of your life, and movie recommendations are integral to your existence. You are tolerant of recommender system, which means you are not easy to exit recommender system even if you have some unsatisfied memory."}

        conformity_dict = { 1:"A Dedicated Follower who gives ratings heavily relies on movie historical ratings, rarely expressing independent opinions. Usually give ratings that are same as historical ratings. ",
                            2:"A Balanced Evaluator who considers both historical ratings and personal preferences when giving ratings to movies. Sometimes give ratings that are different from historical rating.",
                            3:"A Maverick Critic who completely ignores historical ratings and evaluates movies solely based on own taste. Usually give ratings that are a lot different from historical ratings."}
# activity_dict
        diversity_dict = {  1:"An Exceedingly Discerning Selective Viewer who watches movies with a level of selectivity that borders on exclusivity. The movie choices are meticulously curated to match personal taste, leaving no room for even a hint of variety.",
                            2:"A Niche Explorer who occasionally explores different genres and mostly sticks to preferred movie types.",  
                            3:"A Cinematic Trailblazer, a relentless seeker of the unique and the obscure in the world of movies. The movie choices are so diverse and avant-garde that they defy categorization."}
        
        self.conformity_group = init_statistic["conformity"]
        self.activity_group = init_statistic["activity"]
        self.diversity_group = init_statistic["diversity"]
        self.conformity_dsc = conformity_dict[self.conformity_group]
        self.activity_dsc = activity_dict[self.activity_group]
        self.diversity_dsc = diversity_dict[self.diversity_group]

    def init_memory(self):
        """
        Initialize the memory of the avatar
        """
        t1 = time.time()
        def score_normalizer(val: float) -> float:
            return 1 - 1 / (1 + np.exp(val))
        
        embeddings_model = OpenAIEmbeddings(request_timeout = 20)
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {}, relevance_score_fn=score_normalizer)

        LLM = ChatOpenAI(max_tokens=1000, temperature=0.3, request_timeout = 30)
        avatar_retriever = AvatarRetriver(vectorstore=vectorstore, k=5)
        self.memory = AvatarMemory(memory_retriever=avatar_retriever, llm=LLM, reflection_threshold=3, use_wandb = self.use_wandb)
        t2 = time.time()

        
        cprint(f"Avatar {self.avatar_id} is initialized with memory", color='green', attrs=['bold'])
        cprint(f"Time cost: {t2-t1}s", color='green', attrs=['bold'])



    def _reaction(self, messages=None, timeout=30):
        """
        Summarize the feelings of the avatar for recommended item list.
        """ 
        response = ''
        except_waiting_time = 1
        max_waiting_time = 16
        current_sleep_time = 0.5
        while response == '':
            try:
                start_time = time.time()
                time_local = time.localtime(start_time)
                l_start = time.strftime("%Y-%m-%d %H:%M:%S",time_local)

                if(self.use_wandb): # whether to use wandb
                    if((start_time - vars.global_start_time)//vars.global_interval > vars.global_steps):
                        print("\nStart Identifier", start_time, vars.global_start_time, (start_time - vars.global_start_time), vars.global_steps)
                        if(vars.lock.acquire(False)):
                            print("\nStart Identifier", start_time, vars.global_start_time, (start_time - vars.global_start_time), vars.global_steps)
                            vars.global_steps += 1
                            wandb.log(
                                data = {"Real-time Traffic": vars.global_k_tokens - vars.global_last_tokens_record,
                                        "Total Traffic": vars.global_k_tokens,
                                        "Finished Users": vars.global_finished_users,
                                        "Finished Pages": vars.global_finished_pages,
                                        "Error Cast": vars.global_error_cast/1000,
                                },
                                step = vars.global_steps
                            )
                            vars.global_last_tokens_record = vars.global_k_tokens
                            vars.lock.release()
                            print("\nEnd Identifier", time.time(), vars.global_start_time, (time.time() - vars.global_start_time), vars.global_steps)
                            
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", 
                    messages=messages,
                    temperature=0.2,
                    request_timeout = timeout,
                    max_tokens=1000
                    )

                l_end = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
                k_tokens = completion["usage"]["total_tokens"]/1000
                print(f"User {self.avatar_id} used {k_tokens} tokens from {l_start} to {l_end}")
                self.memory.user_k_tokens += k_tokens
                vars.global_k_tokens += k_tokens
                response = completion["choices"][0]["message"]["content"]
            except Exception as e:
                # print(e)
                vars.global_error_cast += 1
                time.sleep(current_sleep_time)
                if except_waiting_time < max_waiting_time:
                    except_waiting_time *= 2
                current_sleep_time = np.random.randint(0, except_waiting_time-1)
                
        return response
    
    
    def make_next_decision(self, remember=False, current_page=None):
        # observation = "How do you feel about the recommender system"
        observation = "Do you satisfy with current recommendation system and what's your interaction history?"
        relevant_memories = self.memory.fetch_memories(observation)
        formated_relevant_memories = self.memory.format_memories_detail(relevant_memories)
        sys_prompt = ("You excel at role-playing. Picture yourself as a user exploring a movie recommendation system. You have the following social traits: " \
                    +f"\nYour activity trait is described as: {self.activity_dsc}"
                    +f"\nNow you are in Page {current_page}. You may get tired with the increase of the pages you have browsed. (above 2 pages is a little bit tired, above 4 pages is very tired)"
                    +f"\nRelevant context from your memory:"
                    +f"\n{formated_relevant_memories}"
                    )
        prompt = ("Firstly, generate an overall feeling based on your memory, in accordance with your activity trait and your satisfaction on recommender system."
                +"\nIf your overall feeling is positive, write: POSITIVE: [reason]"
                +"\nIf it's negative, write: NEGATIVE: [reason]"
                +"\nNext, assess your level of fatigue. You may become tired more easily if you have an inactive activity trait."
                +"\nNow, decide whether to continue browsing or exit the recommendation system based on your overall feeling, activity trait, and tiredness."
                +"\nYou will exit the recommender system either you have negative feelings or you are tired, especially if you have a low activity trait."
                +"\nTo leave, write: [EXIT]; Reason: [brief reason]"
                +"\nTo continue browsing, write: [NEXT]; Reason: [brief reason]"
            )
        messages = [{"role": "system",
                    "content": sys_prompt},
                    {"role": "user",
                    "content": prompt}]
        
        self.write_log("\n" + sys_prompt, color="blue")
        self.write_log("\n" + prompt, color="blue")
        response = self._reaction(messages)
        self.write_log("\n" + response, color="white")

        return response
    
    def response_to_question(self, question, remember=False):
        # prompt_template = ""
        # {agent_summary_description}
        # relevant_memories = self.memory.fetch_memories(question)
        relevant_memories = self.memory.memory_retriever.memory_stream
        formated_relevant_memories = self.memory.format_memories_detail(relevant_memories)
        sys_prompt = (f"You excel at role-playing. Picture yourself as user {self.avatar_id} who has just finished exploring a movie recommendation system. You have the following social traits:"
                +f"\nYour activity trait is described as: {self.activity_dsc}"
                +f"\nYour conformity trait is described as: {self.conformity_dsc}"
                +f"\nYour diversity trait is described as: {self.diversity_dsc}"
                +f"\nBeyond that, your movie tastes are: {'; '.join(self.taste).replace('I ','')}. "
                +"\nThe activity characteristic pertains to the frequency of your movie-watching habits. The conformity characteristic measures the degree to which your ratings are influenced by historical ratings. The diversity characteristic gauges your likelihood of watching movies that may not align with your usual taste."
                )
        prompt = f"""
        Relevant context from user {self.avatar_id}'s memory:
        {formated_relevant_memories}
        Act as user {self.avatar_id}, assume you are having a interview, reponse the following question:
        {question}
        """


        messages = [{"role": "system",
                    "content": sys_prompt},
                    {"role": "user",
                    "content": prompt}]
        
        self.write_log("\n" + sys_prompt, color="blue")
        self.write_log("\n" + prompt, color="blue")
        response = self._reaction(messages)
        self.write_log("\n" + response, color="blue")
        # 
        if(remember):
            self.memory.add_memory(f"I was asked '{question}', and I responsed: '{response}'"
                                , now=datetime.datetime.now())
        return response
    
    def reaction_to_forced_items(self, recommended_items_str):
        """
        Summarize the feelings of the avatar for recommended item list.
        """

        sys_prompt = ("Assume you are a user browsing movie recommendation system who has the following characteristics: "
                # +f"\n{self.conformity_dsc}"
                # +f"\n{self.activity_dsc}"
                # +f"\n{self.diversity_dsc}"
                +f"\nYour movie tastes are: {'; '.join(self.taste).replace('I ','')}. ")
        prompt = (
                "##recommended list## \n" 
                +recommended_items_str
                +"\nPlease choose movies in the ##recommended list## that you want to watch and explain why. After watching the movie, evaluate each movie based on your characteristics, taste and historical ratings to give a rating from 1 to 5."
                +"\nYou only watch movies which aligh with your taste."
                +"\nUse this format: MOVIE: [movie name]; WATCH: [yes or no]; REASON: [brief reason]"
                "\nYou must judge all the movies. If you don't want to watch a movie, use WATCH: no; REASON: [brief reason]"
                +"\nEach response should be on one line. Do not include any additional information or explanations and stay grounded in reality."
        )
        messages = [{"role": "system",
                    "content": sys_prompt},
                    {"role": "user",
                    "content": prompt}]

        reaction = self._reaction(messages, timeout=60)

        return reaction
    
    def reaction_to_recommended_items(self, recommended_items_str, current_page):
        """
        Summarize the feelings of the avatar for recommended item list.
        """ 
        try:
            high_rating = self.high_rating.replace('You are','')
        except:
            # print('low_rating', self.low_rating)
            # print('high_rating', self.high_rating)
            high_rating = ''

        # if(current_page == 1):
        #     self.memory.add_memory(f"My activity characteristic is: {self.activity_dsc}"
        #         , now=datetime.datetime.now())
        #     self.memory.add_memory(f"My conformity characteristic is: {self.conformity_dsc}"
        #         , now=datetime.datetime.now())

        # observation = "What are your activity and conformity characteristic? How do you feel about the recommender system?"
        # relevant_memories = self.memory.fetch_memories(observation)
        # formated_relevant_memories = self.memory.format_memories_detail(relevant_memories)
        top_1_genre = self.user_most_like[self.avatar_id][0]
        taste_str = "I like movies with genres: " + top_1_genre
        sys_prompt = ("You excel at role-playing. Picture yourself as a user exploring a movie recommendation system. You have the following social traits:"
                +f"\nYour activity trait is described as: {self.activity_dsc}"
                +f"\nYour conformity trait is described as: {self.conformity_dsc}"
                +f"\nYour diversity trait is described as: {self.diversity_dsc}"
                +f"\nBeyond that, your movie tastes are: {'; '.join(self.taste).replace('I ','')}. "
                # +f"\nBeyond that, your movie tastes are: {taste_str}. You will watch any movie that contain the genre {top_1_genre}. You won't watch movie that dosn't contain the genre {top_1_genre}."
                # You won't watch movie that dosn't contain the genre {top_1_genre}."
                +f"\nAnd your rating tendency is {high_rating}"#+f"{low_rating}"
                +"\nThe activity characteristic pertains to the frequency of your movie-watching habits. The conformity characteristic measures the degree to which your ratings are influenced by historical ratings. The diversity characteristic gauges your likelihood of watching movies that may not align with your usual taste."
                )
        if self.memory.memory_retriever.memory_stream:
            observation = "What movies have you watched on the previous pages of the current recommender system?"
            relevant_memories = self.memory.fetch_memories(observation)
            formated_relevant_memories = self.memory.format_memories_detail(relevant_memories)
            sys_prompt = sys_prompt +f"\nRelevant context from your memory:{formated_relevant_memories}"

        # prompt = (
        #         "##recommended list## \n" + f"PAGE {current_page}\n"
        #         +recommended_items_str
        #         +"\nPlease response to all the movies in the ##recommended list## and give explains."
        #         +"\nYou will decide the number of movies you want to watch according to your activity characteristic."
        #         +"\nAfter that, assume it's the first time you watch the movie you choose, you will rate movies with 1-5 score reflecting different extent of like, according to your feeling after watching the movie and  your conformity characteristic."
        #         +"\n1 score: not very satisfied with the movie, finding it to have some issues, but with a few redeeming qualities."
        #         +"\n3 score: have many strengths, such as excellent performances, a compelling plot, though there may still be some shortcomings."
        #         +"\n5 score: highly satisfied with the movie, deeming it a masterpiece with outstanding qualities in various aspects Your rating will be influenced by your conformity degree."
        #         +"\nFor response, use this format: MOVIE: [movie name]; WATCH: [yes or no]; REASON: [brief reason]; RATING: [integer between 1-5]; FEELING: [aftermath sentence]; "
        #         +"\nIf you don't want to watch a movie, use WATCH: no; REASON: [brief reason]; RATING: 0; FEELING: [aftermath sentence];"
        #         +"\nFor example: MOVIE: Dracula: Dead and Loving It (1995); WATCH: yes; REASON: comedy is aligned with my movie taste; RATING:2; FEELING: although the genres seem attractive, but the movie quality is poor."
        #         +"\nYou must response to all the recommended movies. Notice that each response must be on one line. Do not include any additional information or explanations and stay grounded in reality."
        # )

        prompt = (
                "#### Recommended List #### \n"
                + f"PAGE {current_page}\n"
                +recommended_items_str
                +"\nPlease respond to all the movies in the ## Recommended List ## and provide explanations."
                +"\nFirstly, determine which movies align with your taste and which do not, and provide reasons. You must respond to all the recommended movies using this format:"
                +"\nMOVIE: [movie name]; ALIGN: [yes or no]; REASON: [brief reason]"
                +"\nSecondly, among the movies that align with your tastes, decide the number of movies you want to watch based on your activity and diversity traits. Use this format:"
                +"\nNUM: [number of movie you choose to watch]; WATCH: [all movie name you choose to watch]; REASON: [brief reason];"
                +"\nThirdly, assume it's your first time watching the movies you've chosen, and rate them on a scale of 1-5 to reflect different degrees of liking, considering your feeling and conformity trait. Use this format:"
                +"\n MOVIE:[movie you choose to watch]; RATING: [integer between 1-5]; FEELING: [aftermath sentence]; "
                +"\n Do not include any additional information or explanations and stay grounded."
        )

        messages = [{"role": "system",
                    "content": sys_prompt},
                    {"role": "user",
                    "content": prompt}]
        
        self.write_log("\n" + sys_prompt, color="blue")
        self.write_log("\n" + prompt, color="blue")
        reaction = self._reaction(messages, timeout=60) # reaction
        self.write_log("\n" + reaction, color="yellow")

        #@ 2 加入用户对这一页用户的满意程度信息

        # =========================
        # pattern = re.compile(r'MOVIE: (.*?) WATCH: (.*?) REASON: (.*?) FEELING: (.*?) RATING: (\d)')
        #pattern = re.compile(r'MOVIE:\s*(.*?)\s*WATCH:\s*(.*?)\s*REASON:\s*(.*?)\s*FEELING:\s*(.*?)\s*RATING:\s*(\d)')
        ###########################################################################################
        # pattern = re.compile(r'MOVIE:\s*(.*?)\s*WATCH:\s*(.*?)\s*REASON:\s*(.*?)\s*RATING:\s*(.*?)\s*FEELING:(.*?)')
        # matches = re.findall(pattern, reaction)
        # all_movies = ", ".join([movie_title.strip(';') for movie_title, watch, reason, rating, feeling in matches])
        # watched_movies = [movie_title.strip(';') for movie_title, watch, reason, rating, feeling in matches if (watch.strip(';').lower() == 'yes')]
        # watched_movies_ratings = [rating.strip(';') for movie_title, watch, reason, rating, feeling in matches if (watch.strip(';').lower() == 'yes')]
        # unwatched_movies = [movie_title.strip(';') for movie_title, watch, reason, rating, feeling in matches if (watch.strip(';').lower() != 'yes')]
        # like_movies = [(movie_title.strip(';'), reason.strip(';'), feeling.strip(';')) for movie_title, watch, reason, rating, feeling in matches if (watch.strip(';').lower() == 'yes' and int(rating.strip(';')) == 5)]
        # dislike_movies = [(movie_title.strip(';'), reason.strip(';'), feeling.strip(';')) for movie_title, watch, reason, rating, feeling in matches if (int(rating.strip(';')) < 4)]
        ###########################################################################################

        # =========================
        pattern1 = re.compile(r'MOVIE: (.+?); RATING: (\d+); FEELING: (.*)')
        match1 = pattern1.findall(reaction)
        pattern2 = re.compile(r'MOVIE: (.+?); ALIGN: (.+?); REASON: (.*)')
        match2 = pattern2.findall(reaction)
        all_movies = ", ".join([movie_title.strip(';') for movie_title, align, reason in match2])
        watched_movies = [movie_title.strip(';') for movie_title, rating, feeling in match1]
        watched_movies_ratings = [rating.strip(';') for movie_title, rating, feeling in match1]
        like_movies = [movie_title.strip(';') for movie_title, rating, feeling in match1 if int(rating.strip(';')) == 5]
        dislike_movies = [movie_title.strip(';') for movie_title, rating, feeling in match1 if (int(rating.strip(';')) < 4)]
        dislike_movies.extend([movie_title.strip(';') for movie_title, align, reason in match2 if align.strip(';').lower() == 'no'])
        self.memory.add_memory(f"The recommender recommended the following movies to me on page {current_page}: {all_movies}, among them, I watched {watched_movies} and rate them {watched_movies_ratings} respectively. I dislike the rest movies: {dislike_movies}."
            , now=datetime.datetime.now()
        )
        # for movie in like_movies:
        #     self.memory.add_memory(f"I like the recommended movie {movie[0]} on page {current_page}, because {movie[1]}, after watching the movie, I feel {movie[2]}"
        #         , now=datetime.datetime.now())
        
        # for movie in dislike_movies:
        #     self.memory.add_memory(f"I dislike the recommended movie {movie[0]} on page {current_page}, because {movie[1]}, after watching the movie, I feel {movie[2]}"
        #         , now=datetime.datetime.now())

        #@ 3 用户做出下一步的决定
        next_decision = self.make_next_decision(current_page=current_page)
        # matches = re.findall(r"\[(.*?)\]", next_decision)[0]
        # print(matches)
        # if(matches == 'EXIT' or matches == 'exit'):
        if('[EXIT]' in next_decision or '[exit]' in next_decision):
            self.exit_flag = True
            self.memory.add_memory(f"After browsing {current_page} pages, I decided to leave the recommendation system."
                , now=datetime.datetime.now())
        
        else:
            self.memory.add_memory(f"Turn to page {current_page+1} of the recommendation."
                , now=datetime.datetime.now())
        #===========================

        #@ 4
        # self.response_to_question("How do you feel about the recommendation result of the recommender?", remember=True)
        # self.response_to_question("Why do you have that feeling?")


        # cprint(f"printing memory {self.avatar_id}", color='green', attrs=['bold'])
        # print(self.memory.memory_retriever.memory_stream)

        # cprint(f"Avatar {self.avatar_id} is exiting", color='green', attrs=['bold'])
        # self.memory.update_memory(reaction) # reflection

        return reaction

    def write_log(self, log, color=None, attrs=None, print=False):
        with open(self.log_file, 'a') as f:
            f.write(log + '\n')
            f.flush()
        if(print):
            cprint(log, color=color, attrs=attrs)
