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
import json
import aiohttp
import time
import logging

import io
openai.api_key = "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"
request_header = {"Authorization": f"Bearer {openai.api_key}"}
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
                df = pd.read_excel(dat,engine='openpyxl')
            else:
                df = pd.read_csv(dat)
            
            # df=pd.DataFrame(data={'text': df, 'n_tokens': df.apply(lambda x: len(tokenizer.encode(x)))})
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

class BaseChoiceHandler():
    file_handler = None
    is_excel = False 
    def _api_endpoint_from_url(self,request_url):
        """Extract the API endpoint from the request URL."""
        return request_url.split("/")[-1]
    
    def _task_id_generator_function(self):
        """Generate integers 0, 1, 2, and so on."""
        task_id = 0
        while True:
            yield task_id
            task_id += 1
    


    def _excel_analysis(self,df, excel_analysis_instructions, model="gpt-3.5-turbo", temperature=0):
        excel_analysis = []
        contexts = df
        
        filedata =[]
        jobs = [{"model": model, "temperature": temperature, "messages": [{"role": "system", "content": "You are a qualitative researcher."}, {"role": "user", "content": f"Categorize the text below into the following categories: {excel_analysis_instructions}\n\n###\n\nText:\n{context}\n\n###\n\nCategory:"}]} for context in contexts]
 
        for job in jobs:
            filedata.append(json.loads(job))  
        response=asyncio.run(self.process_api_requests_from_file(
                requests_data=filedata,
                # save_data=dir+'/summaries_results.jsonl',
                request_url='https://api.openai.com/v1/chat/completions',
                api_key=os.environ.get("OPENAI_API_KEY"),
                max_requests_per_minute=float(3_500 * 0.5),
                max_tokens_per_minute=float(90_000 * 0.5),
                token_encoding_name="cl100k_base",
                max_attempts=int(5),
                logging_level=int(logging.INFO)))
         

        excel_analysis_details=[json.loads(line) for line in response]
        for i in range(len(excel_analysis_details)):
            excel_analysis.append(excel_analysis_details[i][1]['choices'][0]['message']['content'])
        return pd.DataFrame(data={'answer': excel_analysis})

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
    


    def _excel_token_analysis(self,df,file_column,access:bool):         
        excel_total_token_length = 0
        for index, text in df[file_column].items():
            try:
                excel_row_token_length = len(tokenizer.encode(str(text)))
            except:
                excel_row_token_length = 0
                pass
            excel_total_token_length += excel_row_token_length
            if excel_row_token_length > 1500:
                warnings.warn(f'Only the first ~1,000 words of the text \'{text[:100]}...\' in row {str(index+1)} will be recorded and analyzed.')
                truncated_text = text[:len(str(text))*1500//len(tokenizer.encode(str(text)))]
                df[file_column][index] = truncated_text
            if excel_total_token_length > 22500 and not access:
                warnings.warn(f'Only the first {str(index+1)} rows will be recorded and analyzed, as the total word count in the column has exceeded ~15,000 words.')
                df.drop(df.index[index:], inplace=True)
                break
            if excel_total_token_length > 135000 and access:
                warnings.warn(f'Only the first {str(index+1)} rows will be recorded and analyzed, as the total word count in the column has exceeded ~90,000 words.')
                df.drop(df.index[index:], inplace=True)
                break


    def num_tokens_consumed_from_request(self,
        request_json: dict,
        api_endpoint: str,
        token_encoding_name: str,
    ):
        """Count the number of tokens in the request. Only supports chat and embedding requests."""
        encoding = tiktoken.get_encoding(token_encoding_name)
        if api_endpoint == "completions":
            prompt = request_json['messages'][0]['content'] + request_json['messages'][1]['content']
            n = request_json.get("n", 1)
            if isinstance(prompt, str): 
                num_tokens = len(encoding.encode(prompt))
                return num_tokens
            elif isinstance(prompt, list): 
                num_tokens = sum([len(encoding.encode(p)) for p in prompt])
                return num_tokens
            else:
                raise TypeError('Expecting either string or list of strings for "prompt" field in chat request')
        elif api_endpoint == "embeddings":
            input = request_json["input"]
            if isinstance(input, str): 
                try:
                    num_tokens = len(encoding.encode(input)) 
                except:
                    num_tokens = 0
                return num_tokens
            elif isinstance(input, list):
                num_tokens = sum([len(encoding.encode(i)) for i in input])
                return num_tokens
            else:
                raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
        else:
            raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')

class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


class APIRequest:
    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    result = []

    async def call_API(
        self,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_data: list,
        status_tracker: StatusTracker,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=request_url, headers=request_header, json=self.request_json
                ) as response:
                    response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}")
                save_data.append([self.request_json, self.result])
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            save_data.append([self.request_json, response])
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to data")


class ProcessApi(BaseChoiceHandler):
    async def process_api_requests_from_file(self,
            requests_data: str,
            request_url: str,
            api_key: str,
            max_requests_per_minute: float,
            max_tokens_per_minute: float,
            token_encoding_name: str,
            max_attempts: int,
            logging_level: int,
            ):


            seconds_to_pause_after_rate_limit_error = 15
            seconds_to_sleep_each_loop = 0.001 

            # initialize logging
            logging.basicConfig(level=logging_level)
            logging.debug(f"Logging initialized at level {logging_level}")

       
            api_endpoint = self._api_endpoint_from_url(request_url)
            request_header = {"Authorization": f"Bearer {api_key}"}

            # initialize trackers
            queue_of_requests_to_retry = asyncio.Queue()
            task_id_generator = self._task_id_generator_function() 
            status_tracker = StatusTracker()  
            next_request = None

        
            available_request_capacity = max_requests_per_minute
            available_token_capacity = max_tokens_per_minute
            last_update_time = time.time()

            # intialize flags
            file_not_finished = True
            logging.debug(f"Initialization complete.")

            requests_iterator = iter(requests_data)
            while True:
                # get next request (if one is not already waiting for capacity)
                if next_request is None:
                    if queue_of_requests_to_retry.empty() is False:
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
                    elif file_not_finished:
                        try:
                            request_json = eval(next(requests_iterator))
                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=request_json,
                                token_consumption=self.num_tokens_consumed_from_request(request_json, api_endpoint, token_encoding_name),
                                attempts_left=max_attempts,
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(f"Reading request {next_request.task_id}: {next_request}")
                        except StopIteration:

                            logging.debug("Data iterator  exhausted")
                            break   

                # update available capacity
                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
                    max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
                    max_tokens_per_minute,
                )
                last_update_time = current_time
                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if (
                        available_request_capacity >= 1
                        and available_token_capacity >= next_request_tokens
                    ):
                        # update counters
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        # call API
                        response=await asyncio.create_task(
                            next_request.call_API(
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                status_tracker=status_tracker,
                            )
                        )
                        next_request = None
                if status_tracker.num_tasks_in_progress == 0:
                    break
                await asyncio.sleep(seconds_to_sleep_each_loop)

                # if a rate limit error was hit recently, pause to cool down
                seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
                if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
                    remaining_seconds_to_pause = (seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
                    await asyncio.sleep(remaining_seconds_to_pause)
                    # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                    logging.warn(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")

            # after finishing, log final status
            logging.info(f"""Parallel processing complete. Results saved to {response}""")
            if status_tracker.num_tasks_failed > 0:
                logging.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {response}.")
            if status_tracker.num_rate_limit_errors > 0:
                logging.warning(f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")

            return response

            



    

class SumarrizeClass(ProcessApi):
    summary_type:str = ""
    individual_summaries:bool = False
    is_demo = False
    summary_instructions:str = ""
    summary:str = ""
    demo_summary: str = ""

    def __init__(self, is_demo:bool, summary_instructions:str, summary_type:str, individual_summaries:bool) -> None:
        super().__init__()
        self.is_demo = is_demo
        self.data={}
        self.data_demo={}
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
            file_data = []
            jobs = [{"model": model, "temperature": temperature, "messages": [{"role": "system", "content": "You summarize text."}, {"role": "user", "content": f"You summarize text. Using bullet points and punctuation, summarize the text below. {summary_instructions}\n\n###\n\nText:\n{context}\n\n###\n\nSummary (in bullet points):"}]} for context in contexts]
            for job in jobs:
                file_data.append(json.dumps(job))
       
            response=asyncio.run(self.process_api_requests_from_file(
                    requests_data=file_data,
                    # save_data=dir+'/summaries_results.jsonl',
                    request_url='https://api.openai.com/v1/chat/completions',
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    max_requests_per_minute=float(3_500 * 0.5),
                    max_tokens_per_minute=float(90_000 * 0.5),
                    token_encoding_name="cl100k_base",
                    max_attempts=int(5),
                    logging_level=int(logging.INFO)))
         
            summaries_details = [json.loads(line) for line in response]
            for i in range(len(summaries_details)):
                summaries.append(summaries_details[i][1]['choices'][0]['message']['content'])
                n_tokens.append(summaries_details[i][1]['usage']['completion_tokens'])

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
                themes.append("This is your response of your essay.")
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
            filename = []
            jobs = [{"model": model, "temperature": temperature, "messages": [{"role": "system", "content": "You are a qualitative researcher."}, {"role": "user", "content": f"You conduct thematic analysis. Write an extremely detailed essay listing the top themes based on the context below. For each theme, identify quotes and examples (with document source cited), as well as which file(s) contained that theme. {theme_instructions}\n\n###\n\nText:\n{context}\n\n###\n\nDetailed List of Themes (in essay form):"}]} for context in contexts]
 
            for job in jobs:
                filename.append(json.dumps(job))
            resposne=asyncio.run(self.process_api_requests_from_file(
                requests_filepath=dir+'/themes.jsonl',
                # save_filepath=dir+'/themes_results.jsonl',
                request_url='https://api.openai.com/v1/chat/completions',
                api_key=os.environ.get("OPENAI_API_KEY"),
                max_requests_per_minute=float(3_500 * 0.5),
                max_tokens_per_minute=float(90_000 * 0.5),
                token_encoding_name="cl100k_base",
                max_attempts=int(5),
                logging_level=int(logging.INFO)))
    
            themes_details = [json.loads(line) for line in resposne]
            for i in range(len(themes_details)):
                themes.append(themes_details[i][1]['choices'][0]['message']['content'])
                n_tokens.append(themes_details[i][1]['usage']['completion_tokens'])

        return pd.DataFrame(data={'text': themes, 'n_tokens': n_tokens})
    
    def _thematic_analysis(self,df,model="gpt-3.5-turbo", temperature=0):
        contexts = self._aggregate_into_few(df)
        df = self._thematic_analyzer(contexts)


        if len(df['text']) == 1:
            themes = str(df['text'][0])  # Convert 'themes' to a string
            if (themes[-1] != '.' or themes[-1] != '。' or themes[-1] != '।'):
                themes = re.split('(?<=[.。।]) +', themes)
                themes.pop()
                themes = " ".join(themes)
            return themes

        
        # if len(df['text']) == 1:
        #     themes = df['text'][0]
        #     if (themes[-1] != '.' or themes[-1] != '。' or themes[-1] != '।'):
        #        themes = re.split('(?<=[.。।]) +', themes)
        #        themes.pop()
        #        themes = " ".join(themes)
        #     return themes
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
                
            filename = []
            for job in jobs:
                    filename.append(json.dumps(job))
             
            response=asyncio.run(self.process_api_requests_from_file(
                requests_filepath=dir+'/frequencies.jsonl',
                save_filepath=dir+'/frequencies_results.jsonl',
                request_url='https://api.openai.com/v1/chat/completions',
                api_key=os.environ.get("OPENAI_API_KEY"),
                max_requests_per_minute=float(3_500 * 0.5),
                max_tokens_per_minute=float(90_000 * 0.5),
                token_encoding_name="cl100k_base",
                max_attempts=int(5),
                logging_level=int(logging.INFO)))
      
            frequencies_details = [json.loads(line) for line in response]
        
            for i in range(len(frequencies_details)):
                frequencies.append(frequencies_details[i][1]['choices'][0]['message']['content'])
                n_tokens.append(frequencies_details[i][1]['usage']['completion_tokens'])
            
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

    



class ExcelTheme(ThemeAnalysisClass):
    def __init__(self,file_column,theme_type_excel:str='',theme_instructions:str='',acesss:bool=False):
        super().__init__(theme_type=theme_type_excel,theme_instruction=theme_instructions)
        self.file_column=file_column
        self.theme_type_excel=theme_type_excel
        self.theme_instructions=theme_instructions
        self.access=acesss
        self.data={

        }

    def excel_themes(self,df):
        self._excel_token_analysis(df,self.file_column,self.access)
        self.data['themes']=self._thematic_analysis(pd.DataFrame(data={'text':df[self.file_column],'n_tokens':df[self.file_column].apply(lambda x :len(tokenizer.encode(x)))}),self.theme_type_excel,self.theme_instructions)
        self.data['theme_type'] = self.theme_type_excel
        self.data['theme_instructions'] = self.theme_instructions
        self.data['file_column'] = self.file_column
        return self.data


class ExcelCategories(BaseChoiceHandler):
    def __init__(self,file_column,categorize,catgorize_instruction,access:bool=False):
        self.file_column=file_column
        self.categorize_instruction=categorize+ '.' + catgorize_instruction
        self.access=access
        self.data={}


    def excel_categorize(self,df):
        self._excel_token_analysis(df,self.file_column,self.access)
        self.data['categories'] = self._excel_analysis(df[self.file_column], self.categorize_instruction)
        self.data['categorize_instructions'] = self.categorize_instruction
        self.data['column_excel'] = self.file_column
        return self.data
                    

  
            
        
