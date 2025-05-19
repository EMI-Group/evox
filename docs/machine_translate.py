import logging
import re
import sys
import time

import httpx
import polib

logging.basicConfig(level=logging.INFO)

pattern = re.compile(r"^(Bases: )?\{py:(obj|mod)\}`.*?`$")  # match {py:obj}`...` or Bases: {py:obj}`...`

template = """Please translate the following text from a Python evolutionary algorithm library's documentation into Chinese.
- Use simple and clear language.
- For specific terms such as class names and function names (e.g. Algorithms, Problems, jit, API), retain their original English form.
- For python code only segments, please do not translate them, return them as they are.
- For references to academic papers, please do not translate them, return them as they are.
- When the input is too short to give a meaningful translation, please return the original text.
- Maintain the same structured format (e.g. `...`, **...**, (...)[...] block) as the original text.
- Maintain the original links and cross-references.
- Only translate the given text, do not expand or add new content.
- The translate for the following words are provided:
  - algorithm: 算法
  - problem: 问题
  - workflow: 工作流
  - population: 种群
  - evolution: 演化
  - fitness: 适应度

**Only return the translated text; no explanation is needed.**
The text to be translated is: {}"""


class TGIBackend:
    def __init__(
        self,
        base_url: str,
        api_key: str,
    ):
        super().__init__()
        url = "https://" + base_url + "/v1/chat/completions"
        self.url = url
        self.api_key = api_key
        self.num_retry = 10
        self.usage_history = []

    def _one_restful_request(self, query):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
        }
        messages = [
            {
                "role": "user",
                "content": query,
            }
        ]
        data = {
            "messages": messages,
            "stream": False,
            "model": "gpt-4o",
            "max_tokens": 10000,
            "top_p": 0.8,
        }
        for retry in range(self.num_retry):
            try:
                response = httpx.post(self.url, headers=headers, json=data, timeout=30.0)
                json_response = response.json()
            except Exception:
                import traceback

                logging.error(f"Failed to query TGI. Sleep. Retry {retry + 1}...")
                logging.error(traceback.format_exc())
                # network error, sleep and retry
                time.sleep(30)
                continue

            try:
                content = json_response["choices"][0]["message"]["content"]
                usage = json_response["usage"]
                return content, usage
            except Exception:
                logging.error(f"Failed to parse response: {json_response}")
                logging.error(f"{response.text}")
                logging.error(f"{response.json()}")

        logging.error(f"Failed to query TGI for {self.num_retry} times. Abort!")
        raise Exception("Failed to query TGI")

    def query(self, query):
        response = self._one_restful_request(query)

        content, usage = response
        logging.info(f"Received content: {content}")

        return content


if __name__ == "__main__":
    base_url = sys.argv[1]
    api_key = sys.argv[2]
    tgi = TGIBackend(base_url=base_url, api_key=api_key)
    po = polib.pofile("source/locale/zh_CN/LC_MESSAGES/docs.po")
    try:
        for entry in po:
            if entry.msgstr and not entry.fuzzy:
                continue

            if pattern.match(entry.msgid) or entry.msgid.startswith("<svg"):
                logging.info(f"Skipping: {entry.msgid}")
                entry.msgstr = entry.msgid
                continue

            occur_in_tutorial = [("source/tutorial" in filename) for filename, line_num in entry.occurrences]
            occur_in_tutorial = all(occur_in_tutorial)
            if occur_in_tutorial:
                logging.info(f"Skipping: {entry.msgid}")
                entry.msgstr = entry.msgid
                continue

            query = template.format(entry.msgid)
            logging.info(f"Query: {entry.msgid}")
            tranlated = tgi.query(query)
            logging.info("\n")
            entry.msgstr = tranlated
            if entry.fuzzy:
                entry.flags.remove("fuzzy")
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        po.save("source/locale/zh_CN/LC_MESSAGES/docs.po")
