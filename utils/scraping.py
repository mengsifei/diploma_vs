import requests
import BeautifulSoup
import json
import re
import pandas as pd

def scrape_essays(start, end, score):
    link = 'https://writing9.com/band/' + score
    first_level_urls = [link + '/{}'.format(i) for i in range(start, end)]
    all_essays_dict = {"topic": [], "essay": [], "Task Response": [], "Coherence and Cohesion": [], "Lexical Resource": [], "Grammatical Range and Accuracy": []}
    for ind, url in enumerate(first_level_urls):
        first_level_response = requests.get(url)
        first_level_soup = BeautifulSoup(first_level_response.text, 'html.parser')
        links = first_level_soup.find_all('a', href=lambda href: href and href.startswith('/text'))
        first_link_urls = ['https://writing9.com/' + link.get('href') for link in links]
        for i in range(len(first_link_urls)):
            essay_link = first_link_urls[i]
            second_level_response = requests.get(essay_link)
            second_level_soup = BeautifulSoup(second_level_response.text, 'html.parser')
            script = second_level_soup.find('script', {'type': 'application/json'})
            try:
                data = json.loads(script.string)
                if 'props' in data:
                    if 'pageProps' in data['props']:
                        if 'text' in data['props']['pageProps']:
                            question_text = data['props']['pageProps']['text']['question'].replace('\r\n\r\n', '\n')
                            question_text = re.sub(r'\n(\w)', r'\n \1', question_text)
                            question_text = question_text.replace('\n\n', '\n')

                            essay_text = data['props']['pageProps']['text']['text'].replace('\r\n\r\n', '\n')
                            essay_text = re.sub(r'\n(\w)', r'\n \1', essay_text)
                            essay_text = essay_text.replace('\n\n', '\n')
                            bands = data['props']['pageProps']['results']['bands']
                            tr = bands['taBand']
                            cc = bands['coherenceBand']
                            lr = bands['lexicalBand']
                            gra = bands['grammaticBand']
                            all_essays_dict['topic'].append(question_text)
                            all_essays_dict['essay'].append(essay_text)
                            all_essays_dict['Task Response'].append(tr)
                            all_essays_dict['Coherence and Cohesion'].append(cc)
                            all_essays_dict['Lexical Resource'].append(lr)
                            all_essays_dict['Grammatical Range and Accuracy'].append(gra)
            except json.JSONDecodeError:
                continue
    tmp_df = pd.DataFrame(all_essays_dict)
    return tmp_df