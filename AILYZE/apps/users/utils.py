import pandas as pd
import numpy as np
import asyncio
import re
import openai
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
import tiktoken
import fitz
import warnings
from django.conf import settings
import docx2txt
import openpyxl

import io
openai.api_key = "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"
tokenizer = tiktoken.get_encoding("cl100k_base")


class FileHandler:
    max_tokens = 200
    is_excel = False
    files = []
    data = None
    data_binary = True
    data_demo = False
    data_binary_demo = False
    excel = False
    question = False
    summary = False
    themes = False
    frequency = False
    compare_viewpoints_button = False

    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

    def __init__(self,files):
        self.files=files

    def _split_into_many(self,text: str, max_tokens: int = 200) -> list[str]:
        sentences = re.split('(?<=[.。!?।]) +', text)
        n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
        for i, (sentence, token) in enumerate(zip(sentences, n_tokens)):
            if token > max_tokens:
                extra_sentences = sentence.splitlines(keepends=True)
                extra_tokens = [len(tokenizer.encode(" " + extra_sentence)) for extra_sentence in extra_sentences]
                del n_tokens[i]
                del sentences[i]
                n_tokens[i:i] = extra_tokens
                sentences[i:i] = extra_sentences

        chunks = []
        tokens_so_far = 0
        chunk = []
        for i, (sentence, token) in enumerate(zip(sentences, n_tokens)):
            if tokens_so_far + token > max_tokens or (i == (len(n_tokens) - 1)):
                chunks.append(" ".join(chunk))
                chunk = []
                tokens_so_far = 0

            chunk.append(sentence)
            tokens_so_far += token + 1

        return chunks

    def _read_document(self, dat,max_words):
        if dat.name.endswith('.docx') or dat.name.endswith('.doc'):
            text = docx2txt.process(dat)
        else:
            doc = fitz.open(stream=dat.read(), filetype="pdf")
            text = ''
            for page in doc:
                text += page.get_text()
            doc.close()
        text = self.emoji_pattern.sub(r'', text)
        split_text = text.split(' ')
        if len(split_text) > max_words:
            text = " ".join(split_text[:max_words])
            warnings.warn('Only the first '+str(max_words)+' words in the '+str(self.files.name)+' document will be analyzed.')
        if text == '':
            warnings.warn('Warning: The file uploaded contains no content. Please refresh the page and upload another document.')
        return text
    
        
    def process_uploaded_excel(self,dat):
        try:
            if dat.name.endswith('.xls') or dat.name.endswith('.xlsx'):
                df = pd.read_excel(dat)
            else:
                df = pd.read_csv(dat)

            return df
        except Exception as e:
            raise Exception("Something Went wrong ")
                

   

    

    def _process_uploaded_documents(self, max_documents, max_words):
        filenames = []
        texts = []
        for i, dat in enumerate(self.files):
            if i > (max_documents-1):
                print('Only the first '+str(max_documents)+' documents will be analyzed.')
                break
            filenames.append(dat.name)
            text = self._read_document(dat, max_words)
            texts.append(text + '. ')

        df = pd.DataFrame(zip(filenames, texts), columns = ['filename', 'text'])
        df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
        shortened = []
        filenames = []

        for i, row in df.iterrows():
            if row['text'] is None or row['text'] == '':
                continue
            if row['n_tokens'] > self.max_tokens:
                rows = self._split_into_many(row['text'], self.max_tokens)
                rows = ['**Document ' + row['filename'] + ' :**\n\n'+ row_text for row_text in rows]
                filename_repeated = [row['filename']] * len(rows)
                filenames += filename_repeated
                shortened += rows
            else:
                shortened.append(row['text'])
                filenames.append(row['filename'])
            
        df = pd.DataFrame(data = {'text': shortened, 'filename': filenames})
        df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
        self.data = df
        self.data_binary = True
        return df


    def upload_documents(self,max_documents, max_words):
        if self.files :
            for dat in self.files:
                if dat.name.endswith('.xls') or dat.name.endswith('.xlsx') or dat.name.endswith('.csv'):
                    if len(self.files) != 1:
                        raise Exception('Only one Excel file is allowed. Remove all other files except the Excel file.')
                    else:
                        df=self.process_uploaded_excel(dat)
                        print('Note: This feature to analyze spreadsheets is still in beta.')
                        return df
            data=self._process_uploaded_documents(max_documents, max_words)
            return data





class BaseChoiceHandler:
    file_handler = None
    is_excel = False 
    example_questions = ['Explain whether participants trust doctors and substantiate it with examples.',
                     '¿Los participantes toman conscientemente la decisión de vacunarse?',
                     'Where do participants get information about the flu and why?',
                     'Do participants consciously make a decision to vaccinate?',
                     'क्या प्रतिभागी सूचना के स्रोत के रूप में NHS पर भरोसा करते हैं? क्यों?']
    


    # def display_example_questions(self,tips_and_tricks, contact_form):
    #     example_question_buttons = [question for question in self.example_questions]
        
    #     for button, question_demo in zip(example_question_buttons, self.example_questions):
    #         if button:
       
    #             with st.spinner('Identifying document extracts relevant to keywords in your question and analyzing them...'):
    #                 display_precomputed_example_answer(question_demo, tips_and_tricks, contact_form)

    def _to_excel(df):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
            writer.close()
        excel_data = buffer.getvalue()
        return excel_data

    def _to_txt(text):
        data_str = text
        buffer = io.BytesIO(data_str.encode())
        return buffer.getvalue()
    
    def _get_frequency(context, frequency_viewpoint, model="gpt-3.5-turbo", temperature=0):
        if settings.ENVIRONMENT !='dev':
            response = openai.ChatCompletion.create(model=model,
                                                    temperature=temperature,
                                                    messages=[{"role": "system", "content": "You are a qualitative researcher."},
                                                            {"role": "user", "content": f"Based on the text below, answer in the form of a table with 3 columns, namely (a) Document Name; (b) Contains Viewpoint? - Yes/No/Maybe; and (c) Detailed Description. Focus on whether it can be inferred that the document contains the following viewpoint: {frequency_viewpoint}.\n\n###\n\nText:\n{context}\n\n###\n\nTable:"}]
                                                )
            response = response['choices'][0]['message']['content'].strip()
            return response
        else:
            response="this is your frequency value"
            return response


    def _covert_to_df(markdown):
        rows = [row.strip().split("|")[1:-1] for row in markdown.strip().split('\n') if row.strip()]
        df = pd.DataFrame(rows, columns=rows[0])
        return df



    def _aggregate_into_few(self,df, join = ' '):
        aggregated_text = []
        current_text = ''
        current_length = 0
        max_index = len(df['n_tokens']) - 1
        token_length = sum(df['n_tokens'])
        if token_length > 3096:
            max_length = round(token_length / np.ceil(token_length / 3096)) + 100
        else:
            max_length = 3096
        if max_index == 0:
            return df['text']
        for i, row in df.iterrows():
            current_length += row['n_tokens']
            if current_length > max_length:
                current_length = 0
                aggregated_text.append(current_text)
                current_text = ''

            current_text = current_text + join + row['text']
            if max_index == i:
                aggregated_text.append(current_text)
        return aggregated_text
    

    def create_context(self,keywords, df, max_len, size):
        q_embeddings = openai.Embedding.create(input=keywords, engine='text-embedding-ada-002')['data'][0]['embedding']
        df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

        returns = []
        cur_len = 0
        for i, row in df.sort_values('distances', ascending=True).iterrows():
            cur_len += row['n_tokens'] + 4
            if cur_len > max_len:
                break
            returns.append(row["text"])
        return "\n\n===\n\n".join(returns)
    
    def longest_str_intersection(self,a: str, b: str) -> str:
        seqs = [a[pos1:pos2 + 1] for pos1 in range(len(a)) for pos2 in range(len(a))]
        seqs = [seq for seq in seqs if seq != '']
        max_len_match = 0
        max_match_sequence = ''
        for seq in seqs:
            if seq in b and len(seq) > max_len_match:
                max_len_match = len(seq)
                max_match_sequence = seq
        return max_match_sequence
    





    # def display_precomputed_example_answer(self,question_demo, tips_and_tricks, contact_form):
    #     precomputed_answers = {
    #         'Explain whether participants trust doctors and substantiate it with examples.': (difference_in_views_answer, difference_in_views_top_context, difference_in_views_context),
    #         '¿Los participantes toman conscientemente la decisión de vacunarse?': (spanish_answer, spanish_top_context, spanish_context),
    #         'Where do participants get information about the flu and why?': (information_about_flu_answer, information_about_flu_top_context, information_about_flu_context),
    #         'Do participants consciously make a decision to vaccinate?': (conscious_decision_answer, conscious_decision_top_context, conscious_decision_context),
    #         'क्या प्रतिभागी सूचना के स्रोत के रूप में NHS पर भरोसा करते हैं? क्यों?': (hindi_answer, hindi_top_context, hindi_context)
    #     }
    

class SumarrizeClass(BaseChoiceHandler):
    summary_type:str = ""
    individual_summaries:bool = False
    is_demo = False
    summary_instructions:str = ""
    summary:str = ""
    demo_summary: str = ""

    def __init__(self, is_demo:bool, summary_instructions:str, summary_type:str, individual_summaries:bool) -> None:
        super().__init__()
        # self.user = user
        self.is_demo = is_demo
        # self.file_handler = file_handler
        self.data={}
        self.data_demo={}
        # self.is_excel = file_handler.is_excel
        self.summary_type = summary_type
        self.summary_instructions = summary_instructions
        self.individual_summaries = individual_summaries

  

    def _summarizer(self, contexts, summary_type, summary_instructions, model="gpt-3.5-turbo", temperature=0):
        summaries = []
        n_tokens = []

        if len(contexts) == 1 and summary_type == 'Essay':
            if settings.ENVIRONMENT != "dev":
                response = openai.ChatCompletion.create(model=model,
                                                        temperature=temperature,
                                                        messages=[{"role": "system", "content": "You summarize text."},
                                                                {"role": "user", "content": f"Write an extremely detailed essay summarising the context below. {summary_instructions}\n\n###\n\nContext:\n{contexts[0]}\n\n###\n\nDetailed Summary (in essay form):"}]
                )
          

                summaries.append(response['choices'][0]['message']['content'].strip())
                n_tokens.append(len(tokenizer.encode(response['choices'][0]['message']['content'].strip())))

            elif len(contexts) == 1 and summary_type == 'Bullet points':
                response = openai.ChatCompletion.create(model=model,
                                                        temperature=temperature,
                                                        messages=[{"role": "system", "content": "You summarize text."},
                                                                {"role": "user", "content": f"Using bullet points, write an extremely detailed summary of the context below. {summary_instructions}\n\n###\n\nContext:\n{contexts[0]}\n\n###\n\nDetailed Summary (in bullet points):"}]
                )

                summaries.append(response['choices'][0]['message']['content'].strip())
                n_tokens.append(len(tokenizer.encode(response['choices'][0]['message']['content'].strip())))
            else:
                summaries.append(" your got the answer")
                n_tokens.append(" these are your tokens")

        else:
            pass
            # filename = dir + "/summaries.jsonl"
            # jobs = [{"model": model, "temperature": temperature, "messages": [{"role": "system", "content": "You summarize text."}, {"role": "user", "content": f"You summarize text. Using bullet points and punctuation, summarize the text below. {summary_instructions}\n\n###\n\nText:\n{context}\n\n###\n\nSummary (in bullet points):"}]} for context in contexts]
            # with open(filename, "w") as f:
            #     for job in jobs:
            #         json_string = json.dumps(job)
            #         f.write(json_string + "\n")
            # asyncio.run(process_api_requests_from_file(
            #     requests_filepath=dir+'/summaries.jsonl',
            #     save_filepath=dir+'/summaries_results.jsonl',
            #     request_url='https://api.openai.com/v1/chat/completions',
            #     api_key=os.environ.get("OPENAI_API_KEY"),
            #     max_requests_per_minute=float(3_500 * 0.5),
            #     max_tokens_per_minute=float(90_000 * 0.5),
            #     token_encoding_name="cl100k_base",
            #     max_attempts=int(5),
            #     logging_level=int(logging.INFO)))
            # with open(dir+'/summaries_results.jsonl', 'r') as f:
            #     summaries_details = [json.loads(line) for line in f]
            # if os.path.exists(dir+'/summaries_results.jsonl'):
            #     os.remove(dir+'/summaries_results.jsonl')
            # if os.path.exists(dir+'/summaries.jsonl'):
            #     os.remove(dir+'/summaries.jsonl')
            # for i in range(len(summaries_details)):
            #     summaries.append(summaries_details[i][1]['choices'][0]['message']['content'])
            #     n_tokens.append(summaries_details[i][1]['usage']['completion_tokens'])

        return pd.DataFrame(data={'text': summaries, 'n_tokens': n_tokens})
    

    def summarize(self,df, summary_type, summary_instructions, model="gpt-3.5-turbo", temperature=0):
        contexts = self._aggregate_into_few(df)
        df = self._summarizer(contexts, summary_type, summary_instructions)
        if len(df['text']) == 1:
            summary = df['text'][0]
            if summary_type == "Essay" and (summary[-1] != '.' or summary[-1] != '。' or summary[-1] != '।'):
                summary = re.split('(?<=[.。।]) +', summary)
                summary.pop()
                summary = " ".join(summary)
            return summary
        else:
            df = self.summarize(df, summary_type, summary_instructions)
            return df
    def call_summarize(self, data, demo=False):
        if self.summary_instructions == '':
            self.summary_instructions = 'Include headers (e.g., introduction, conclusion) and specific details.'
        if not self.individual_summaries:
            if demo:
                self.data_demo['summary_type'] = self.summary_type
                self.data_demo['summary_instructions'] = self.summary_instructions
                self.data_demo['summary'] = self.summarize(
                    data,
                    summary_type=str(self.data_demo['summary_type'][0]),
                    summary_instructions=str(self.data_demo['summary_instructions'][0])
                ).strip()   
            else:
                self.data['summary_type'] = self.summary_type
                self.data['summary_instructions'] = self.summary_instructions
                self.data['summary'] = self.summarize(
                    data,
                    summary_type=self.summary_type,
                    summary_instructions=self.summary_instructions
                ).strip()
        if demo:
            summary = self.data_demo['summary']
        else:
            summary = self.data['summary']
        return self.data




class QuestionClass(BaseChoiceHandler):
    question:str=''
    is_check:bool=False
    instruction:str=''
    keywords:str=''

    def __init__(self,question = "Does the participant trust doctors and why?",keywords = "doctors, patients, trust, confident, believe",instruction=''):
        self.question=question+' '+ instruction
        self.keywords=keywords+' '+question
        self.data={}
        
    def answer_question(self,df, get_quotes = False, model="gpt-3.5-turbo", max_len=3097, size="ada", debug=False, temperature=0):
        keywords=self.keywords
        context = self.create_context(keywords, df, max_len, size)
        all_context = ''
        for line in context.splitlines():
            all_context = all_context + str(line) + '\n'
        if debug:
            print("Context:\n" + context)
            print("\n\n")
        try:
            if settings.ENVIRONMENT != "dev":
                response = openai.ChatCompletion.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": f"Write an essay to answer the question below, based on a plausible inference of the document extracts below. \n\n###\n\nDocument extracts:\n{context}\n\n###\n\nQuestion:\n{self.question}\n\n###\n\nEssay:"}]
                )
                response = response['choices'][0]['message']['content'].strip() 
            else:
                response="this is yor resposne for yor specific question"        
            if not get_quotes:
                quotes = 'x'
                all_context = 'xy'
            
            else: 
                raw_quotes = openai.ChatCompletion.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": f"You are a quote extractor. Only take quotes from the document extracts below. For each quote, use in-text citations to cite the document source. In bullet points, substantiate the argument below by providing quotes.\n\n###\n\nArgument (don't take quotes from here):\n{response}\n\n###\n\nDocument extracts (take quotes from here):\n{context}\n\n###\n\nQuotes in bullet points (with document source cited for each quote):"}]
                )

                raw_quotes = raw_quotes['choices'][0]['message']['content'].strip()
                raw_quotes = raw_quotes.replace('$', '\$')
                all_context = all_context.replace('$', '\$').replace('[', '\[')
                quotes = ''
                for quote in raw_quotes.split("\n"):
                    if len(quote) > 50:
                        raw_quote = self.longest_str_intersection(quote, all_context)
                        quotes = quotes + '* "' + raw_quote + '"\n'
                        all_context = all_context.replace(raw_quote, ':green['+raw_quote+']')

            answer_context= [response, quotes, all_context]
            self.data['answer']=answer_context[0]
            self.data['context']=answer_context[1]
            self.data['all_context']=answer_context[2]
            return self.data
        
        except Exception as e:
            print(e)
            return ""

    
class ThemeAnalysisClass(BaseChoiceHandler):
    theme_type:str=''
    theme_instructions:str=''

    def __init__(self,theme_type, theme_instruction):
        self.theme_type=theme_type
        self.theme_instructions=theme_instruction
        self.data={}


    def _thematic_analyzer(self,contexts, model="gpt-3.5-turbo", temperature=0):
        themes = []
        n_tokens = []

     
            
        if len(contexts) == 1 and self.theme_type == 'Essay':
            if settings.ENVIRONMENT != "dev":
                response = openai.ChatCompletion.create(model=model,
                                                        temperature=temperature,
                                                        messages=[{"role": "system", "content": "You are a qualitative researcher."}, {"role": "user", "content": f"You conduct thematic analysis. Write an extremely detailed essay identifying the top 10 themes, based on the context below. For each theme, specify which file(s) contained that theme and include as many quotes and examples as possible. {self.theme_instructions}\n\n###\n\nContext:\n{contexts[0]}\n\n###\n\nEssay (excluding introduction and conclusion):"}]
                )

                themes.append(response['choices'][0]['message']['content'].strip())
                n_tokens.append(len(tokenizer.encode(response['choices'][0]['message']['content'].strip())))
            else:
                themes.append("This is your response of your essay")
                n_tokens.append("This is your tokens value ")

        elif len(contexts) == 1 and self.theme_type == 'Codebook':
            if settings.ENVIRONMENT != "dev":
                response = openai.ChatCompletion.create(model=model,
                                                        temperature=temperature,
                                                        messages=[{"role": "system", "content": "You are a qualitative researcher."}, {"role": "user", "content": f"You conduct thematic analysis. Based on the context below, create a codebook in the form of a table with three columns, namely (a) theme; (b) detailed description with as many quotes and examples as possible (with document source cited); and (c) file which contains that theme. {self.theme_instructions}\n\n###\n\nContext:\n{contexts[0]}\n\n###\n\nCodebook (in table format):"}]
                )

                themes.append(response['choices'][0]['message']['content'].strip())
                n_tokens.append(len(tokenizer.encode(response['choices'][0]['message']['content'].strip())))
            else:
                themes.append("This is your response of your code book")
                n_tokens.append("This is your tokens value ")

        else:
            pass
            # filename = dir + "/themes.jsonl"
            # jobs = [{"model": model, "temperature": temperature, "messages": [{"role": "system", "content": "You are a qualitative researcher."}, {"role": "user", "content": f"You conduct thematic analysis. Write an extremely detailed essay listing the top themes based on the context below. For each theme, identify quotes and examples (with document source cited), as well as which file(s) contained that theme. {theme_instructions}\n\n###\n\nText:\n{context}\n\n###\n\nDetailed List of Themes (in essay form):"}]} for context in contexts]
            # with open(filename, "w") as f:
            #     for job in jobs:
            #         json_string = json.dumps(job)
            #         f.write(json_string + "\n")
            # asyncio.run(process_api_requests_from_file(
            #     requests_filepath=dir+'/themes.jsonl',
            #     save_filepath=dir+'/themes_results.jsonl',
            #     request_url='https://api.openai.com/v1/chat/completions',
            #     api_key=os.environ.get("OPENAI_API_KEY"),
            #     max_requests_per_minute=float(3_500 * 0.5),
            #     max_tokens_per_minute=float(90_000 * 0.5),
            #     token_encoding_name="cl100k_base",
            #     max_attempts=int(5),
            #     logging_level=int(logging.INFO)))
            # with open(dir+'/themes_results.jsonl', 'r') as f:
            #     themes_details = [json.loads(line) for line in f]
            # if os.path.exists(dir+'/themes_results.jsonl'):
            #     os.remove(dir+'/themes_results.jsonl')
            # if os.path.exists(dir+'/themes.jsonl'):
            #     os.remove(dir+'/themes.jsonl')
            # for i in range(len(themes_details)):
            #     themes.append(themes_details[i][1]['choices'][0]['message']['content'])
            #     n_tokens.append(themes_details[i][1]['usage']['completion_tokens'])

        return pd.DataFrame(data={'text': themes, 'n_tokens': n_tokens})
    def _thematic_analysis(self,df, model="gpt-3.5-turbo", temperature=0):
        contexts = self._aggregate_into_few(df)
        df = self._thematic_analyzer(contexts)
        
        if len(df['text']) == 1:
            themes = df['text'][0]
            if (themes[-1] != '.' or themes[-1] != '。' or themes[-1] != '।'):
               themes = re.split('(?<=[.。।]) +', themes)
               themes.pop()
               themes = " ".join(themes)
            return themes
        else:
            df = self._thematic_analysis(df, self.theme_type, self.theme_instructions)
            return df    
        

    def check_theme_type(self,df):
        if self.theme_type=='Codebook':
            theme_df=self._covert_to_df(self._thematic_analysis(df))
            theme_data=self._to_excel(theme_df)
        else:
            theme_data=self._to_txt(self._thematic_analysis(df))
        self.data['theme_type']=self.theme_type
        self.data['theme_instruction']=self.theme_instructions
        self.data['answer']=theme_data
        return self.data

                    

      


      
class FrequencyHandlerClass(SumarrizeClass):
    frequency_viewpoint:str=''

    def __init__(self, frequency_viewpoint:str) -> None:
        super().__init__(is_demo=False,summary_instructions=frequency_viewpoint,summary_type='Essay',individual_summaries=False)
        self.frequency_viewpoint=frequency_viewpoint
        self.data={}

    def frequency_analyzer(self,df, model="gpt-3.5-turbo", temperature=0):
        filenames = df['filename'].unique()
        jobs = []
        frequencies = []
        n_tokens = []
        
        if settings.ENVIRONMENT !='dev':

            for filename in filenames:
                question_frequency_viewpoint = "Based on the context below, write a short essay explaining using examples if the document " + filename + " contains the following viewpoint: " + self.frequency_viewpoint + "."
                context = self.create_context(self.frequency_viewpoint, df.loc[df['filename'] == filename], max_len=3097, size="ada")
                jobs += [{"model": model, "temperature": temperature, "messages": [{"role": "system", "content": "You are a qualitative researcher."}, {"role": "user", "content": f"{question_frequency_viewpoint}\n\n###\n\nContext:\n{context}\n\n###\n\nShort Essay:"}]}]
                
            # filename = dir + "/frequencies.jsonl"
            # with open(filename, "w") as f:
            #     for job in jobs:
            #         json_string = json.dumps(job)
            #         f.write(json_string + "\n")
            # asyncio.run(process_api_requests_from_file(
            #     requests_filepath=dir+'/frequencies.jsonl',
            #     save_filepath=dir+'/frequencies_results.jsonl',
            #     request_url='https://api.openai.com/v1/chat/completions',
            #     api_key=os.environ.get("OPENAI_API_KEY"),
            #     max_requests_per_minute=float(3_500 * 0.5),
            #     max_tokens_per_minute=float(90_000 * 0.5),
            #     token_encoding_name="cl100k_base",
            #     max_attempts=int(5),
            #     logging_level=int(logging.INFO)))
            # with open(dir+'/frequencies_results.jsonl', 'r') as f:
            #     frequencies_details = [json.loads(line) for line in f]
            # if os.path.exists(dir+'/frequencies_results.jsonl'):
            #     os.remove(dir+'/frequencies_results.jsonl')
            # if os.path.exists(dir+'/frequencies.jsonl'):
            #     os.remove(dir+'/frequencies.jsonl')
            # for i in range(len(frequencies_details)):
            #     frequencies.append(frequencies_details[i][1]['choices'][0]['message']['content'])
            #     n_tokens.append(frequencies_details[i][1]['usage']['completion_tokens'])
            
            return pd.DataFrame(data={'text': frequencies, 'n_tokens': n_tokens})
        else:
            frequencies="frequency values"
            n_tokens="frequency tokens"
            return pd.DataFrame(data={'text': frequencies, 'n_tokens': n_tokens})


    def frequency_analysis(self,df, model="gpt-3.5-turbo", temperature=0):
        contexts = self._aggregate_into_few(df, join = '\n\n')
        if len(contexts) == 1:
            frequency = self._get_frequency(contexts[0], self.frequency_viewpoint, model, temperature)
            return frequency

        else:
            df = self._summarizer(contexts, summary_type = 'Essay', summary_instructions = 'Focus on whether the documents contained the following viewpoint: ' + self.frequency_viewpoint + '.')
            if len(df['text']) == 1:
                frequency = self._get_frequency(df['text'][0], self.frequency_viewpoint, model, temperature)
                return frequency
            else:
                df = self.frequency_analysis(df)
                return df
            
    def call_frequency(self,df):
        response=self.frequency_analysis(self.frequency_analyzer(df))
        self.data['frequency_Instruction']=self.frequency_viewpoint
        self.data['answer']=response
        return self.data
        
class CompareViewPointsClass(BaseChoiceHandler):
    question_compare_groups:str=''
    instruction_compare_groups:str=''
    get_quotes:bool=False
    instructions_only:str=''
    keywords_only:str=''


    def __init__(self,question_compare_groups = "Does the participant trust doctors and why?",keywords_only = "doctors, patients, trust, confident, believe", instruction_compare_groups='',instructions_only='')-> None:
        super().init()
        self.question=question_compare_groups +' '+instruction_compare_groups+' '+instructions_only
        self.keywords=keywords_only+' '+question_compare_groups
        self.data={}
     


    def answer_question(self,df,  get_quotes = False, model="gpt-3.5-turbo", max_len=3097, size="ada", debug=False, temperature=0):
        keyqords=self.keywords
        context = self.create_context(keyqords, df, max_len=max_len, size=size)
        all_context = ''
        for line in context.splitlines():
            all_context = all_context + str(line) + '\n'
        if debug:
            print("Context:\n" + context)
            print("\n\n")
        try:
            response = openai.ChatCompletion.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": f"Write an essay to answer the question below, based on a plausible inference of the document extracts below. \n\n###\n\nDocument extracts:\n{context}\n\n###\n\nQuestion:\n{self.question}\n\n###\n\nEssay:"}]
            )
            response = response['choices'][0]['message']['content'].strip()
            if not get_quotes:
                quotes = 'x'
                all_context = 'xy'
            else: 
                raw_quotes = openai.ChatCompletion.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": f"You are a quote extractor. Only take quotes from the document extracts below. For each quote, use in-text citations to cite the document source. In bullet points, substantiate the argument below by providing quotes.\n\n###\n\nArgument (don't take quotes from here):\n{response}\n\n###\n\nDocument extracts (take quotes from here):\n{context}\n\n###\n\nQuotes in bullet points (with document source cited for each quote):"}]
                )

                raw_quotes = raw_quotes['choices'][0]['message']['content'].strip()
                raw_quotes = raw_quotes.replace('$', '\$')
                all_context = all_context.replace('$', '\$').replace('[', '\[')

                quotes = ''

                for quote in raw_quotes.split("\n"):
                    if len(quote) > 50:
                        raw_quote = self.longest_str_intersection(quote, all_context)
                        quotes = quotes + '* "' + raw_quote + '"\n'
                        all_context = all_context.replace(raw_quote, ':green['+raw_quote+']')
                
            answer_context= [response, quotes, all_context]
            self.data['answer']=answer_context[0]
            self.data['context']=answer_context[1]
            self.data['all_context']=answer_context[2]
            return self.data

        except Exception as e:
            print(e)
            return ""

    





