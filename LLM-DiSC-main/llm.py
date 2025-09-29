import importlib.util
import json
import os
import time

from openai import OpenAI
from functools import cached_property
from pathlib import Path
import tempfile
import re
from env import INFO
import datetime

# Define the log file path
LOG_FILE_PATH = 'llm_interaction_log.txt'
LOG_TOKEN_PATH = 'llm_token_log.txt'
conversation_history = []
file_path = 'controller.py'
function_name = 'get_control'
debug_count = 0
tokens_count = 0
first_run = True
run_directory = None
initial_prompt_generated = False
LLM_temp = 1.0

# Make sure to complete the information before running.
# LLM_model = ''
# LLM_api = ''
# LLM_url = ''
def create_run_directory():
    """
    Create a folder to store all relevant files from each run

    :return: Folder path
    """
    global run_directory
    if run_directory is None:
        run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        run_directory = os.path.join('runs\experiments', run_id)
        os.makedirs(run_directory, exist_ok=True)
    return run_directory


def log_interaction(run_directory, conversation_history):
    """
    Record interaction history
    """
    global first_run
    log_file_path = os.path.join(run_directory, LOG_FILE_PATH)

    # If it is the first run, write the startup information.
    if first_run:
        with open(log_file_path, 'a', encoding='utf-8') as file:
            file.write(f"{'-' * 100}\n")
            file.write(f"Starting interaction at {datetime.datetime.now()}\n")
            file.write(f"model: {LLM_model}\n")
            file.write(f"temperature: {LLM_temp}\n")
            file.write(f"{'-' * 100}\n")
            first_run = False

    with open(log_file_path, 'r', encoding='utf-8') as file:
        existing_content = file.readlines()


    startup_info_end_index = existing_content.index(f"{'-' * 100}\n", 1) + 1
    startup_info = existing_content[:startup_info_end_index]


    with open(log_file_path, 'w', encoding='utf-8') as file:
        file.writelines(startup_info)  # 写入启动信息
        for interaction in conversation_history:
            role = interaction["role"]
            content = interaction["content"]
            file.write(f"{role.capitalize()}:\n{content}\n\n")


def get_response(conversation_history, api_key= LLM_api,
                 url=LLM_url):

    client = OpenAI(api_key=api_key, base_url=url)
    retries = 0
    max_retries = 3
    retry_delay = 10
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=LLM_model,
                messages=conversation_history,
                temperature=LLM_temp
            )
            print('get response')
            tokens = response.usage.total_tokens
            return response.choices[0].message.content, tokens
        except Exception as e:
            print(f"Error occurred: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying... ({retries}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print("Max retries reached, unable to fetch response.")
                raise


def log_token_count(run_directory, tokens):

    global tokens_count
    log_file_path = os.path.join(run_directory, LOG_TOKEN_PATH)
    tokens_count += tokens
    with open(log_file_path, 'a', encoding='utf-8') as file:
        file.write(f" Tokens used this call: {tokens}, Total tokens used so far: {tokens_count} " + "\n")


def log_run_time(run_directory, time):
    log_file_path = os.path.join(run_directory, LOG_TOKEN_PATH)
    with open(log_file_path, 'a', encoding='utf-8') as file:
        file.write(f" total execution time: {time}s" + "\n")

def log_success(run_directory, run_time, steps, trajectory_lengths,iterations):
    log_file_path = os.path.join(run_directory, LOG_TOKEN_PATH)
    trajectory_sum = sum(trajectory_lengths)
    num = len(trajectory_lengths)
    average_trajectory_length = trajectory_sum / num
    with open(log_file_path, 'a', encoding='utf-8') as file:
        file.write(f"steps of every agent: {steps}" + "\n")
        file.write(f"trajectory length: {trajectory_lengths}" + "\n")
        file.write(f"total trajectory length: {trajectory_sum}" + "\n")
        file.write(f"average trajectory length: {average_trajectory_length}" + "\n")
        file.write(f"controller run time: {run_time}s" + "\n")
        file.write(f"iterations: {iterations}" + "\n")


def get_code(response):
    """
    get code from response
    """
    response = repr(response)
    code_blocks = re.findall(r'```python(.*?)```', response, re.DOTALL)
    if code_blocks:
        code = code_blocks[0].replace("\\n", "\n")
        code = re.sub(r"\\(['\"])", r'\1', code)
        return code
    else:
        return None


def run_python_code(code):
    """
    Run the Python code and check for errors.
    """
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return True, None
    except Exception as e:
        print(f"Error: {e}")
        return False, str(e)


def load_function_from_file():
    """
    Load functions from the .py file
    """
    try:
        spec = importlib.util.spec_from_file_location(function_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        function = getattr(module, function_name)
        return function
    except Exception as e:
        print(f"Error: {e}")
        return None



def get_initial_prompt():

    info = INFO()
    user_prompt = fr"""
There are `{info.agent_num}` homogeneous agents, each with a radius of `{info.agent_radius}`. The agents follow the dynamics $\dot{{\boldsymbol{{x}} }} =f(\boldsymbol{{x}}) + g(\boldsymbol{{x}}) \boldsymbol{{u}}$, where $\boldsymbol{{x}} = [p_x, p_y,\theta]$ represents the agent's position $(p_x,p_y)$ and orientation $\theta$ in a 2D space, and $\boldsymbol{{u}} = [v, \omega]$ represents the control inputs, with $v$ as the linear velocity and $\omega$ as the angular velocity. The initial states of the agents are`{info.begin_state}`, with their initial controls are `{info.begin_ctrl}`. Each agent has a known target point at `{info.target_pos}` (stop moving when the agent reaches the target point or gets sufficiently close to it). The positions of the obstacles are given as `{info.poly_pos}`, each sublist represents the vertices of a polygon, with each vertex stored in the form of `(x, y)`. The last vertex is the same as the first, indicating a closed polygon.

Please help design a controller and implement it as Python function. The function should output the control input signal needed to update the agent's state. The time interval for the agent's state update is `{info.dt}`. The function should return the control signal. Please design the controller using the following Python function structure and do not need the `example usage`:

```
def {function_name}(index,all_state,all_ctrl,target_pos,obs_pos,agent_R, state_info):
        # Your code to compute the control input
        return control_input
```
Where:

- 'index' is the agent index, starting from 0.
- 'all_state' is the current state of all the agents in the format [[x1, y1,theta1], [x2, y2,theta2], ...].
- 'all_ctrl' is the control signal of the agent in the format [[v1, omega1], [v2, omega2], ...].
- 'target_pos' is the target position of the agent in the format [x, y].
- 'obs_pos' is a list containing the vertex coordinates of multiple polygons, formatted as [[(x1,y1),(x2,y2),...,(x1,y1)],[...],...]. Each sublist represents the vertices of a polygon, the last vertex is the same as the first, indicating a closed polygon.
- 'agent_R' is the radius of the agent.
- 'state_info' is the state change of all agents from the initial moment to the current moment, formatted as [[[x11, y11,theta11], [x12, y12,theta12],...], [[x21, y21,theta21], [x22, y22,theta22],...],...]. (For example, `state_info[index]` records the state change of the agent `index` from the initial moment to the current moment.)

"""
    system_prompt = f"""
You are an expert in designing controllers. Your task is to: 
- Analyze the given prompt and design a controller for the agent based on the provided information. 
- Implement the controller in Python, adhering to the specific format given in the prompt. 
- Ensure the controller enables the agent to avoid obstacles and other agents while successfully reaching its target.
"""
    return user_prompt, system_prompt


def main():
    global debug_count, run_directory, initial_prompt_generated, tokens_count
    run_directory = create_run_directory()
    if not initial_prompt_generated:
        initial_prompt, system_prompt = get_initial_prompt()
        conversation_history.append({"role": "system", "content": system_prompt})
        conversation_history.append({"role": "user", "content": initial_prompt})
        initial_prompt_generated = True
    # initial_prompt = get_initial_prompt()
    # conversation_history.append({"role": "user", "content": initial_prompt})
    while True:
        response, tokens = get_response(conversation_history)
        log_token_count(run_directory, tokens)
        conversation_history.append({"role": "assistant", "content": response})
        python_code = get_code(response)
        if python_code:

            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(python_code)

            code_file_path = os.path.join(run_directory, file_path)
            with open(code_file_path, 'w', encoding='utf-8') as code_file:
                code_file.write(python_code)

            success, error = run_python_code(python_code)
            if error:
                debug_count += 1
                conversation_history.append({"role": "user", "content": f"Error encountered: {error}"})
                log_interaction(run_directory, conversation_history)
            else:
                print("Code executed successfully!")
                log_interaction(run_directory, conversation_history)
                break
        else:
            error = "No Python code found in response."
            print(error)
            conversation_history.append({"role": "user", "content": error})
            log_interaction(run_directory, conversation_history)

    return run_directory


if __name__ == '__main__':
    main()
