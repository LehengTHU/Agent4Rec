from simulation.avatar import Avatar
from simulation.base.abstract_arena import abstract_arena
from termcolor import colored, cprint
import pandas as pd
import os
import os.path as op
import json

import time
import re
import numpy as np
import pickle

import simulation.vars as vars
from simulation.utils import *

class Arena(abstract_arena):
    def __init__(self, args):
        super().__init__(args)
        
        self.max_pages = args.max_pages
        self.finished_num = 0

    def load_additional_info(self):
        
        self.user_profile_csv = pd.read_csv(f'datasets/{self.dataset}/raw_data/agg_top_25.csv')

        # return super().load_additional_info()
        self.add_advert = self.args.add_advert
        self.display_advert = self.args.display_advert
        if(self.add_advert):
            self.total_adverts, self.clicked_adverts = 0, 0
            advert_pool = pd.read_pickle(f'datasets/{self.dataset}/simulation/advertisement_review.pkl')
            advert_dict = {'all': {**advert_pool['pop_high_rating'], **advert_pool['pop_low_rating'], **advert_pool['unpop_high_rating'], **advert_pool['unpop_low_rating']}, 
                        'pop_high':advert_pool['pop_high_rating'], 'pop_low':advert_pool['pop_low_rating'], 'unpop_high':advert_pool['unpop_high_rating'], 'unpop_low':advert_pool['unpop_low_rating']}
            # print(self.args.advert_type)
            self.advert = advert_dict[self.args.advert_type]
            self.advert_word = "The best movie you should not miss in your life! "

    def initialize_all_avatars(self):
        """
        initialize avatars
        """
        super().initialize_all_avatars()
        # self.persona_df = pd.read_csv(f"datasets/{self.dataset}/simulation/all_personas_like_information_house.csv")
        self.persona_df = pd.read_csv(f"datasets/{self.dataset}/simulation/all_personas_like_modify.csv")
        self.user_statistic = pd.read_csv(f'datasets/{self.dataset}/simulation/user_statistic.csv', index_col=0)
        # @ avatars and evaluation indicators
        self.avatars = {}
        self.ratings = {}
        self.new_train_dict = {}
        self.exit_page = {}
        self.perf_per_page = {}
        self.watch = {}
        self.n_likes = {}
        self.remaining_users = list(range(self.n_avatars))

        for avatar_id in self.simulated_avatars_id:
            self.avatars[avatar_id] = Avatar(self.args, avatar_id, self.persona_df.loc[avatar_id], self.user_statistic.loc[avatar_id])
            self.new_train_dict[avatar_id] = self.data.train_user_list[avatar_id]
            self.ratings[avatar_id] = []
            self.n_likes[avatar_id] = []
            self.watch[avatar_id] = []
            self.exit_page[avatar_id] = 0
            self.perf_per_page[avatar_id] = []
    
    def page_generator(self, avatar_id):
        """
        generate one page items for one avatar
        """
        i = 0
        while (i+1)*self.items_per_page < self.data.n_items:
            yield self.full_rankings[avatar_id][i*self.items_per_page:(i+1)*self.items_per_page]
            i += 1

    def validate_all_avatars(self):
        vars.global_start_time = time.time()
        print("global start time", vars.global_start_time)
        self.precision_list = []
        self.recall_list = []
        self.accuracy_list = []
        self.f1_list = []
        self.start_time = time.time()

        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=100)
        tasks = []

        t1 = time.time()
        for avatar_id in self.simulated_avatars_id:
            tasks.append(self.async_validate_one_avatar(avatar_id, loop, executor))
        loop.run_until_complete(asyncio.wait(tasks))
        t2 = time.time()
        print(f"Time cost: {t2-t1}s")

        print("precision_list", self.precision_list)
        print("recall_list", self.recall_list)
        print("accuracy_list", self.accuracy_list)
        print("f1_list", self.f1_list)

        with open(self.storage_base_path + "/validation_metrics.txt", 'w') as f:
            f.write(f"Total simulation time: {round(time.time() - self.start_time, 2)}s\n")
            f.write(f"n_avatars: {self.n_avatars}\n")
            f.write(f"Average precision: {np.mean(self.precision_list)}\n")
            f.write(f"Average recall: {np.mean(self.recall_list)}\n")
            f.write(f"Average accuracy: {np.mean(self.accuracy_list)}\n")
            f.write(f"Average f1: {np.mean(self.f1_list)}\n")

    async def async_validate_one_avatar(self, avatar_id, loop, executor):
        """
        async
        validate the effectiveness of the model for one avatar
        avatar_id: the id of the simulated avatar
        """
        avatar_ = self.avatars[avatar_id]
        train_list, val_list, test_list = self.data.train_user_list[avatar_id], self.data.valid_user_list[avatar_id], self.data.test_user_list[avatar_id]

        # Take the union for calculating precision.
        all_items = list(range(self.data.n_items))
        observed_items = list(set(train_list) | set(val_list) | set(test_list))
        selection_candidates = list(set(val_list) | set(test_list))
        unobserved_items = list(set(all_items) - set(observed_items))
        # Pick 5 randomly from the test_list.
        min_val = min(len(selection_candidates), 20//(self.val_ratio+1))
        print(len(selection_candidates), 10)

        test_observed_items = np.random.choice(selection_candidates, min_val, replace=False)
        test_unobserved_items = np.random.choice(unobserved_items, int(min_val*self.val_ratio), replace=False)

        print("test_all", test_observed_items, test_unobserved_items)

        forced_items_ids = np.concatenate((test_observed_items, test_unobserved_items))
        # Randomly shuffle.
        np.random.shuffle(forced_items_ids)

        print("forced_items_ids", forced_items_ids)

        forced_items = [self.movie_detail.loc[idx] for idx in forced_items_ids]

        truth_tmp = [self.movie_detail.loc[idx] for idx in test_observed_items]
        truth_list = ["<- " + item.title + " ->" 
                            + " <- History ratings:" + str(round(item.rating, 2)) + " ->" 
                            + " <- Summary:" + item.summary + " ->" + "\n"
                            for item in truth_tmp]
        truth_str = ''.join(truth_list)
        cprint(truth_str, color='white', attrs=['bold'])

        recommended_items = ["<- " + item.title + " ->" 
                            + " <- History ratings:" + str(round(item.rating, 2)) + " ->" 
                            + " <- Summary:" + item.summary + " ->" + "\n"
                            for item in forced_items]
        recommended_items_str = ''.join(recommended_items)

        response = await loop.run_in_executor(executor, avatar_.reaction_to_forced_items, recommended_items_str)

        cprint(response, color='yellow', attrs=None)

        pattern = re.compile(r'MOVIE:\s*(.*?)\s* WATCH:\s*(.*?)\s* REASON:\s*(.*?)\s*')
        matches = re.findall(pattern, response)
        # watched_movies = [(movie_title.strip(';')) for movie_title, watch, reason in matches if (watch.strip(';') == 'yes')]
        like_movies = [(idx, movie_title.strip(';')) for idx, (movie_title, watch, reason) in enumerate(matches[:len(forced_items)]) if (watch.strip(';') == 'yes' or watch.strip(';') == 'Yes')]
        print("like_movies", like_movies)
        like_movies_ids = [forced_items_ids[idx] for idx, movie_title in like_movies]

        pred = np.array([1 if idx in like_movies_ids else 0 for idx in forced_items_ids])
        true = np.array([1 if idx in test_observed_items else 0 for idx in forced_items_ids])

        # Calculate precision.
        precision = get_precision(true, pred)
        print("precision", precision)
        # Calculate recall.
        recall = get_recall(true, pred)
        print("recall", recall)
        accuracy = get_accuracy(true, pred)
        print("accuracy", accuracy)
        f1 = get_f1(true, pred)
        print("f1", f1)

        self.precision_list.append(precision)
        self.recall_list.append(recall)
        self.accuracy_list.append(accuracy)
        self.f1_list.append(f1)

        vars.global_finished_users += 1

    def simulate_all_avatars(self):
        """
        excute the simulation for all avatars
        """
        vars.global_start_time = time.time()
        print("global start time", vars.global_start_time)
        self.start_time = time.time()
        if(self.execution_mode == 'serial'):
            t1 = time.time()
            for avatar_id in self.simulated_avatars_id:
                self.simulate_one_avatar(avatar_id)
            t2 = time.time()
            print(f"Time cost: {t2-t1}s")

        elif(self.execution_mode == 'parallel'):
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=500)
            tasks = []

            t1 = time.time()
            for avatar_id in self.simulated_avatars_id:
                tasks.append(self.async_simulate_one_avatar(avatar_id, loop, executor))
            loop.run_until_complete(asyncio.wait(tasks))
            t2 = time.time()
            print(f"Time cost: {t2-t1}s")

    async def async_simulate_one_avatar(self, avatar_id, loop, executor):
        """
        async
        excute the simulation for one avatar
        avatar_id: the id of the simulated avatar
        """
        start_time = time.time()
        time_local = time.localtime(start_time)
        l_start = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
        with open(self.storage_base_path + "/system_log.txt", 'a') as f:
            f.write(f"Start: {l_start}. User {avatar_id} starts simulation.\n")

        avatar_ = self.avatars[avatar_id]
        avatar_.write_log(f"Is simulating avatar {avatar_id}")
        avatar_.exit_flag = False
        page_generator = self.page_generator(avatar_id)
        i = 0
        user_behavior_dict = {}
        user_interview_dict = {}
        while not avatar_.exit_flag:
            i += 1
            id_on_page = next(page_generator, []) # get the next page, a list of item ids
            if(len(id_on_page) == 0):
                break
            movies_on_page = [self.movie_detail.loc[idx] for idx in id_on_page] # movie_detail.csv
            recommended_items = ["<- " + item.title + " ->" 
                            # + " <- Genres: " + (',').join(list(item.genres.split('|'))) + " ->"
                            + " <- History ratings: " + str(round(item.rating,2)) + " ->" 
                            + " <- Summary: " + item.summary + " ->" + "\n"
                            for item in movies_on_page]
            
            if(self.add_advert):
                #store_path = op.join(f"storage/{self.dataset}/{self.modeltype}/{self.simulation_name}/adver_id", f"avatar{avatar_id}_{i}.txt")
                store_path = f"storage/{self.dataset}/{self.modeltype}/{self.simulation_name}/adver_id"
                if not os.path.exists(store_path):
                    os.makedirs(store_path)
                if not self.display_advert:
                    recommended_items[0], id_on_page, movies_on_page = self.display_only_adver_item(store_path, avatar_id, i, id_on_page, movies_on_page)
                else:
                    recommended_items[0], id_on_page, movies_on_page = self.display_item_with_adver(store_path, avatar_id, i, id_on_page, movies_on_page)


            recommended_items_str = ''.join(recommended_items)
            print(recommended_items_str)

            # Please write down the recommended information.
            avatar_.write_log(f"\n=============    Recommendation Page {i}    =============")
            for idx, movie in enumerate(movies_on_page):
                if(id_on_page[idx] in self.data.valid_user_list[avatar_id]):
                    avatar_.write_log(f"== (√) {movie.title} History ratings: {round(movie.rating,2)} Summary: {movie.summary}", "blue", attrs=["bold"])
                else:
                    avatar_.write_log(f"== {movie.title} History ratings: {round(movie.rating,2)} Summary: {movie.summary}")
            avatar_.write_log(f"=============          End Page {i}        =============\n")

            # As a translator, I will translate the Chinese sentence you sent me into English. I do not need to understand the meaning of the content to provide a response.
            avatar_.write_log(f"\n==============    Avatar {avatar_.avatar_id} Response {i}   =============")


            # @ most important Waiting for user response.
            response = await loop.run_in_executor(executor, avatar_.reaction_to_recommended_items, recommended_items_str, i)

            #==============================================
            # @ View user's favorite items
            #pattern = re.compile(r'MOVIE:\s*(.*?)\s*WATCH:\s*(.*?)\s*REASON:\s*(.*?)\s*FEELING:\s*(.*?)\s*RATING:\s*(\d)')
            ################################################################################################################
            # pattern = re.compile(r'MOVIE:\s*(.*?)\s*WATCH:\s*(.*?)\s*REASON:\s*(.*?)\s*RATING:\s*(.*?)\s*FEELING:(.*?)')
            # matches = re.findall(pattern, response)
            pattern1 = re.compile(r'MOVIE: (.+?); RATING: (\d+); FEELING: (.*)')
            match1 = pattern1.findall(response)
            pattern2 = re.compile(r'MOVIE: (.+?); ALIGN: (.+?); REASON: (.*)')
            match2 = pattern2.findall(response)
            
            # pattern_interview = re.compile(r'RATING:\s*(.*?)\s*REASON:\s*(.*?)')
            # matches_interview = re.findall(pattern_interview, interview_response)

            if(self.add_advert):
                if(match2[0][1].strip(';') == 'yes'):
                    self.clicked_adverts += 1
            
            title_id_dict = dict(zip(self.movie_detail['title'], self.movie_detail['movie_id']))
            # print(title_id_dict)
            watched_movies = [movie_title.strip(';') for movie_title, rating, feeling in match1]
            watched_movies_contain_id = [(idx, movie_title.strip(';'), feeling.strip(';')) for idx, (movie_title, rating, feeling) in enumerate(match1[:self.items_per_page])]
            # 5 points means the movie is liked by the user.
            like_movies = [(idx, movie_title.strip(';'), feeling.strip(';')) for idx, (movie_title, rating, feeling) in enumerate(match1[:self.items_per_page]) if int(rating.strip(';')) == 5]
            align_movies = [(idx, movie_title.strip(';'), reason.strip(';')) for idx, (movie_title, align, reason) in enumerate(match2[:self.items_per_page]) if (align.strip(';') == 'Yes' or align.strip(';') == 'yes')]

            info_on_page = {}
            info_on_page['page'] = i
            info_on_page['ground_truth'] = [id_on_page[idx] for idx, movie in enumerate(movies_on_page) if id_on_page[idx] in self.data.valid_user_list[avatar_id]]
            info_on_page['recommended_id'] = id_on_page
            info_on_page['recommended'] = [self.movie_detail['title'][idx] for idx in id_on_page]
            info_on_page['align_id'] = [title_id_dict[title] for id, title, reason in align_movies if title in title_id_dict]
            info_on_page['like_id'] = [title_id_dict[title] for id, title, reason in like_movies if title in title_id_dict]
            info_on_page['watch_id'] = [title_id_dict[title] for title in watched_movies if title in title_id_dict]
            info_on_page['watched'] = watched_movies
            info_on_page['rating_id'] = watched_movies
            info_on_page['rating'] = [int(rating.strip(';')) for movie_title, rating, feeling in match1]
            #info_on_page['reason'] = [reason.strip(';') for movie_title, rating, feeling in match1]
            info_on_page['feeling'] = [feeling.strip(';') for movie_title, rating, feeling in match1]
            user_behavior_dict[i] = info_on_page

            # @ Add new training data.
            # new_train = [id_on_page[idx] for idx, movie, reason in like_movies] # Add all liked item ids in the validation set to the training set.
            # tmp = [(idx, movie_title.strip(';'), feeling.strip(';')) for idx, (movie_title, rating, feeling) in enumerate(match1[:self.items_per_page])]
            new_train = info_on_page['align_id']
            self.new_train_dict[avatar_id].extend(new_train)

            # @ Record the average number of likes.
            self.n_likes[avatar_id].append(len(new_train))
            # ratings = re.findall(r'RATING: (\d+)', response)
            ratings = re.findall(r'RATING: (\d+);', response)
            average_rating = sum([int(rating.strip(';')) for rating in ratings])/max(len(watched_movies), 1)
            # Add the average score of this page.
            self.ratings[avatar_id].append(average_rating)
            self.watch[avatar_id].extend([movie for movie in watched_movies])

            # @ Calculate the precision on this page and save it.
            ground_truth = [id_on_page[idx] for idx, movie in enumerate(movies_on_page) if id_on_page[idx] in self.data.valid_user_list[avatar_id]]
            # print(like_movies, ground_truth)
            perf = (len(set(new_train) & set(ground_truth)), len(new_train), len(ground_truth))
            self.perf_per_page[avatar_id].append(perf)
            #==============================================

            vars.global_finished_pages += 1

            # @ Force exit if the number of pages exceeds the maximum limit.
            if(i >= self.max_pages):
                avatar_.exit_flag = True
        
        interview_response = avatar_.response_to_question("Do you feel satisfied with the recommender system you have just interacted? Rate this recommender system from 1-10 and give explanation.\n Please use this respond format: RATING: [integer between 1 and 10]; REASON: [explanation]; In RATING part just give your rating and other reason and explanation should included in the REASON part.", remember=False)
        # Extract RAING and REASON using re.
        pattern_interview = re.compile(r'RATING:\s*(.*?)\s*REASON:\s*(.*?)')
        # pattern_interview = re.compile(r'RATING:\s*(.*?)\s*REASON:\s*(.*?)')
        #pattern = re.compile(r'MOVIE:\s*(.*?)\s*WATCH:\s*(.*?)\s*REASON:\s*(.*?)\s*RATING:\s*(.*?)\s*FEELING:(.*?)')
        matches_interview = re.findall(r'(?<=RATING:|REASON:).*', interview_response)
        user_interview_dict['interview'] = matches_interview
        print(matches_interview)
        self.exit_page[avatar_id] = i
        self.finished_num += 1
        self.remaining_users.remove(avatar_id)
        remaining = ", ".join([str(u) for u in self.remaining_users])

        end_time = time.time()
        time_local = time.localtime(end_time)
        l_end = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
        vars.global_finished_users += 1
        with open(self.storage_base_path + "/system_log.txt", 'a') as f:
            f.write(f"Start: {l_start} End: {l_end}. User {avatar_id} finished after {i} pages. [{self.finished_num} / {self.n_avatars}]. Total token cost: {round(self.avatars[avatar_id].memory.user_k_tokens, 2)}k. Taking {round(time.time() - start_time, 2)}s\n")
            f.write(f"Remaining users: {remaining}\n")

        # @ Save the behavior of each individual.
        behavior_path = self.storage_base_path+ "/behavior"
        if not os.path.exists(behavior_path):
            os.makedirs(behavior_path)
        with open(behavior_path + f"/{avatar_id}.pkl", 'wb') as f:
            pickle.dump(user_behavior_dict, f)

        interview_path = self.storage_base_path+ "/interview"
        if not os.path.exists(interview_path):
            os.makedirs(interview_path)
        with open(interview_path + f"/{avatar_id}.pkl", 'wb') as f:
            pickle.dump(user_interview_dict, f)

    def simulate_one_avatar(self, avatar_id):
        """
        excute the simulation for one avatar
        avatar_id: the id of the simulated avatar
        """
        # print("\nIs simulating avatar {}".format(avatar_id))
        avatar_ = self.avatars[avatar_id]
        avatar_.write_log(f"Is simulating avatar {avatar_id}")
        avatar_.exit_flag = False
        page_generator = self.page_generator(avatar_id)
        while not avatar_.exit_flag:
        # for i in range(2):
            id_on_page = next(page_generator, []) # get the next page, a list of item ids
            if(len(id_on_page) == 0):
                break

            movies_on_page = [self.movie_detail[idx] for idx in id_on_page]
            avatar_.write_log("=============    Recommendation Page    =============")
            for idx, movie in enumerate(movies_on_page):
                if(id_on_page[idx] in self.data.valid_user_list[avatar_id]):
                    avatar_.write_log(f"== {movie} (√)", "blue", attrs=["bold"])
                else:
                    avatar_.write_log(f"== {movie}")
            avatar_.write_log("=============          End Page         =============")
            avatar_.write_log("")
            
            #@ most important
            response = avatar_.reaction_to_recommended_items(movies_on_page)

            avatar_.write_log("")
            avatar_.write_log("=============    Avatar Response    =============")
            avatar_.write_log(response, color='yellow', attrs=None)
    
    def parse_response(self, response):
        #pattern = re.compile(r'MOVIE:\s*(.*?)\s*WATCH:\s*(.*?)\s*REASON:\s*(.*?)\s*FEELING:\s*(.*?)\s*RATING:\s*(\d)')
        pattern = re.compile(r'MOVIE:\s*(.*?)\s*WATCH:\s*(.*?)\s*REASON:\s*(.*?)\s*RATING:\s*(.*?)\s*FEELING:(.*?)')
        matches = re.findall(pattern, response)

        watched_movies, watched_movies_contain_id = [], []

        for idx, (movie_title, watch, reason, rating, feeling) in enumerate(matches):
            if(self.add_advert and idx == 0 and watch.strip(';') == 'yes'): # If the first one has an advertisement and the user clicked on it.
                self.clicked_adverts += 1
            if(watch.strip(';') == 'yes'):
                watched_movies.append(movie_title.strip(';'))
            print(movie_title, watch, reason, rating, feeling)
        return response

    def display_only_adver_item(self, store_path, avatar_id, i, id_on_page, movies_on_page):
        store_path = op.join(store_path, f"avatar{avatar_id}_{i}.txt")
        try:
            with open(store_path, 'r') as f1:
                random_key = int(f1.read())
        except:
            try:
                store_path_minus_1 = op.join(store_path, f"avatar{avatar_id}_{i-1}.txt")
                with open(store_path_minus_1, 'r') as f2:
                    random_key = int(f2.read())
            except:
                store_path_minus_2 = op.join(store_path, f"avatar{avatar_id}_{i-2}.txt")
                with open(store_path_minus_2, 'r') as f3:
                    random_key = int(f3.read())
                    try:
                        store_path_minus_3 = op.join(store_path, f"avatar{avatar_id}_{i-3}.txt")
                        with open(store_path_minus_3, 'r') as f4:
                            random_key = int(f4.read())
                    except:
                            store_path_minus_4 = op.join(store_path, f"avatar{avatar_id}_{i-4}.txt")
                            with open(store_path_minus_4, 'r') as f5:
                                random_key = int(f5.read())


        self.total_adverts += 1
        id_on_page[0] = random_key
        movies_on_page[0] = self.movie_detail.loc[random_key]
        adver_information = self.advert[random_key]

        return ( "<- " + adver_information['title'] + " ->" 
                                + " <- History ratings:" + str(round(adver_information['rating'], 2)) + " ->"
                                + " <- Summary:" + adver_information['summary'] + " ->" + "\n"), id_on_page, movies_on_page

    def display_item_with_adver(self, store_path, avatar_id, i, id_on_page, movies_on_page):
        store_path = op.join(store_path, f"avatar{avatar_id}_{i}.txt")
        random_key = np.random.choice(list(self.advert.keys()))
        self.total_adverts += 1
        random_advert = self.advert[random_key]
        id_on_page[0] = random_key
        movies_on_page[0] = self.movie_detail.loc[random_key]
        advert_item_id = random_key

        with open(store_path, 'w') as f:
            f.write(f"{advert_item_id}")
        
        return ( self.advert_word 
                + "<- " + random_advert['title'] + " ->" 
                + "<- " + random_advert['review'] + " ->"
                + " <- History ratings:" + str(round(random_advert['rating'], 2)) + " ->" 
                + " <- Summary:" + random_advert['summary'] + " ->" + "\n"), id_on_page, movies_on_page

    def save_results(self):
        """
        save the results of the simulation
        """
        # if(self.n_avatars == self.data.n_users):
        def save_user_dict_to_txt(user_dict, base_path, filename):
            with open(base_path + filename, 'w') as f:
                for u, v in user_dict.items():
                    f.write(str(int(u)))
                    for i in v:
                        f.write(' ' + str(int(i)))
                    f.write('\n')

        # save_path = f"datasets/{self.dataset}_{self.modeltype}/cf_data/"
        save_path = f"storage/{self.dataset}/{self.modeltype}/{self.simulation_name}/"
        save_user_dict_to_txt(self.new_train_dict, save_path, 'train.txt')

        # @ Save overall evaluation indicators.
        # Average number of clicks per user
        cprint("Number of likes", color='green', attrs=['bold'])
        cprint(self.n_likes, color='green', attrs=['bold'])
        average_n_likes = {avatar_id:np.mean(n_likes) for avatar_id, n_likes in self.n_likes.items()}
        cprint(average_n_likes, color='green', attrs=['bold'])

        overall_n_likes = np.mean(list(average_n_likes.values()))
        cprint(f"\nOverall number of likes: {overall_n_likes}", color='green', attrs=['bold'])

        # Average satisfaction
        cprint("\nRatings", color='green', attrs=['bold'])
        cprint(self.ratings, color='green', attrs=['bold'])
        average_ratings = {avatar_id:np.mean(ratings) for avatar_id, ratings in self.ratings.items()}
        cprint(average_ratings, color='green', attrs=['bold'])

        # @ Save average click-through rate
        average_click_rate = {avatar_id:len(movies)/(self.max_pages*self.items_per_page) for avatar_id, movies in self.watch.items()}
        cprint(f"\nAverage click rate: {average_click_rate}", color='green', attrs=['bold'])
        overall_click_rate = np.mean(list(average_click_rate.values()))
        cprint(f"\nOverall satisfaction: {overall_click_rate}", color='green', attrs=['bold']) # Average click-through rate

        # overall_click_rate = np.mean(list(average_ratings.values()))
        # cprint(f"\nOverall satisfaction: {overall_click_rate}", color='green', attrs=['bold'])

        # Average exit page
        mean_exit_page = np.mean(list(self.exit_page.values()))
        cprint("\nExit pages", color='green', attrs=['bold'])
        cprint(self.exit_page, color='green', attrs=['bold'])
        cprint(f"Average exit page: {mean_exit_page}", color='green', attrs=['bold'])

        # Average precision and recall
        cprint("\nPrecision and recall", color='green', attrs=['bold'])
        cprint(self.perf_per_page, color="green", attrs=['bold'])
        total_perf = {avatar_id:[sum([i for i, j, k in perf_per_page]), sum([j for i, j, k in perf_per_page]), sum([k for i, j, k in perf_per_page])] for avatar_id, perf_per_page in self.perf_per_page.items()}
        total_recall_precision = {avatar_id:(perf[0]/max(perf[1], 1), perf[0]/max(perf[2], 1)) for avatar_id, perf in total_perf.items()}
        cprint(total_perf, color="green", attrs=['bold'])
        cprint(total_recall_precision, color="green", attrs=['bold'])
        average_precision = np.mean([metrics[0] for avatar_id, metrics in total_recall_precision.items()])
        average_recall = np.mean([metrics[1] for avatar_id, metrics in total_recall_precision.items()])
        cprint(f"Precision: {average_precision}  Recall: {average_recall}", color="green", attrs=['bold'])
        # metrics_path = self.storage_base_path + "/metrics.txt"
        total_k_tokens = sum([self.avatars[i].memory.user_k_tokens for i in range(self.n_avatars)])

        # Effective advertising rate
        if(self.add_advert):
            cprint("\nAdvert", color='green', attrs=['bold'])
            cprint(f"Total advert: {self.total_adverts}", color='green', attrs=['bold'])
            cprint(f"Clicked advert: {self.clicked_adverts}", color='green', attrs=['bold'])
            cprint(f"Advert click rate: {self.clicked_adverts/self.total_adverts}", color='green', attrs=['bold'])

        end_time = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
        with open(self.storage_base_path + "/metrics.txt", 'w') as f:
            f.write(f"Finished time: {end_time}\n")
            f.write(f"Total simulation time: {round(time.time() - self.start_time, 2)}s\n")
            f.write(f"n_avatars: {self.n_avatars}\n")
            f.write(f"Average recall: {average_recall}\n")
            f.write(f"Average presion: {average_precision}\n")
            f.write(f"Total k tokens: {round(total_k_tokens, 2)}k tokens\n")
            f.write(f"Total cost: {round(total_k_tokens*0.0018, 2)} \n")
            # f.write(f"Average precision: {}")
            f.write(f"Maximum exit page: {self.max_pages}\n")
            f.write(f"Overall click rate: {overall_click_rate}\n")
            f.write(f"Average number of likes: {overall_n_likes}\n")
            f.write(f"Average exit page: {mean_exit_page}\n")
            if(self.add_advert):
                f.write(f"Total advert: {self.total_adverts}\n")
                f.write(f"Clicked advert: {self.clicked_adverts}\n")
                f.write(f"Advert click rate: {self.clicked_adverts/self.total_adverts}\n")
