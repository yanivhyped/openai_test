# Install the required libraries
#!pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client

# Import the necessary libraries
import os
import time
import requests
import json
import google.auth
from google.oauth2.credentials import Credentials
from google.cloud import language_v1
import pandas as pd
from google.cloud import bigquery
import requests
from lxml import html

site_config={'cheezburger.com':'mu-container mu-content','cracked.com':'page-content'}



def get_page_content(url,path):
    # Send a request to the URL
    r = requests.get(url)

    # Parse the HTML content
    tree = html.fromstring(r.content)

    # Find the element with the class "page-content"
    page_content = tree.xpath('//*[@class="{}"]'.format(path))[0]

    # Return the content inside the element
    return page_content.text_content()


def run(site,top):
    # Set the project, dataset, and table where the results will be stored
    project_id = 'your-project-id'
    dataset_id = 'dbt_cdoyle'
    table_id = 'discover_entities'
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/yanivbenatia/nlp_test/nlp.json'
    # Set the credentials to use for the API calls
    #credentials = Credentials.from_authorized_user_info(info=None)
    credentials, project_id = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

    # Set the client for the Natural Language API
    client = language_v1.LanguageServiceClient(credentials=credentials)


    # Set the client for the BigQuery API
    bq_client = bigquery.Client(project=project_id, credentials=credentials)

    # Query the top 20 URLs from the source table, ordered by clicks
    query = """
    SELECT url, sum(clicks) as clicks
    FROM `literally-analytics.rivery.search_console_discover_by_url`
    WHERE property_id = 'sc-domain:{}'
    and date > '2022-01-01'
    group by 1
    ORDER BY 2 DESC
    LIMIT {}
    """.format(site,top)
    query_job = bq_client.query(query)
    urls = query_job.to_dataframe()

    # Initialize empty lists to store the results
    categories = []
    entities=[]
    categories_url_list = []
    categories_clicks_list=[]
    entities_url_list = []
    entities_clicks_list=[]
    categories_sites=[]
    entities_sites=[]



    # Iterate through the URLs and extract the categories
    for index, row in urls.iterrows():
        url = row['url']
        clicks = row['clicks']
        document = language_v1.Document(
            content= get_page_content(url,site_config[site]),
            type=language_v1.Document.Type.HTML
            )
        
        # response = client.classify_text(request={"document":document})
        # categories.extend([category.name for category in response.categories])
        # categories_url_list.extend([url] * len(response.categories))
        # categories_clicks_list.extend([clicks] * len(response.categories))
        # categories_sites.extend([site] * len(response.categories))
        #   document = language_v1.Document(
        #       content= ' '.join(url.split('-')[1::]),
        #       type=language_v1.Document.Type.PLAIN_TEXT
        #      )
        response = client.analyze_entities(request={"document":document})
        entities.extend([[token.name,token.type_,token.salience] for token in response.entities[::20]])
        entities_url_list.extend([url] * len(response.entities))
        entities_clicks_list.extend([clicks] * len(response.entities))
        entities_sites.extend([site] * len(response.entities))

        #   response = client.analyze_syntax(request={"document":document,'encoding_type':language_v1.EncodingType.UTF8})
        #   topics.extend([token.text.content for token in response.tokens])
        #   url_list.extend([url] * len(response.tokens))
        #   clicks_list.extend([clicks] * len(response.tokens))
        print(url)
        time.sleep(1)

    # Create a dataframe with the results
    #df = pd.DataFrame({'category': categories, 'topic': topics, 'url': url_list, 'clicks': clicks_list})


    # Create a dataframe with the results for categories 
    #categories_df = pd.DataFrame({'categories': categories, 'url': categories_url_list, 'clicks': categories_clicks_list,'site':categories_sites})


    # Create a dataframe with the results for categories 
    entities_df = pd.DataFrame({'entities': entities, 'url': entities_url_list, 'clicks': entities_clicks_list,'site':entities_sites})

    # Write the results to the destination table
    #bq_client.load_table_from_dataframe(categories_df, f'{project_id}.{dataset_id}.{table_id}').result()
    bq_client.load_table_from_dataframe(entities_df, f'{project_id}.{dataset_id}.{table_id}').result()

run('cracked.com',40)