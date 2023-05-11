from headlineAI import Fetcher


fetcher = Fetcher('knowyourmeme.com', 50, 0,platform='guides',content_type='guides',model='headline_kym_guides')

#fetcher.insert_result_to_db(filepath='knowyourmeme.com_collections_on_chz_module.csv')
#fetcher.generate_training_dataset()
fetcher.run_generative_ai_on_headlines()