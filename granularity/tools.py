import time
import openai

# Define a function that adds a delay to a Completion API call
def delayed_completion(delay_in_seconds: float = 1, max_trials: int = 1, **kwargs):
    """Delay a completion by a specified amount of time."""

    # Sleep for the delay
    time.sleep(delay_in_seconds)

    # Call the Completion API and return the result
    output, error = None, None
    for _ in range(max_trials):
        try:
            output = openai.ChatCompletion.create(**kwargs)
            break
        except Exception as e:
            error = e
            pass
    return output, error

def post_process(completion):
    content = completion['choices'][0]['message']['content'].strip()
    result = []
    if 'Yes' in content and 'No' not in content:
        result.append('Yes')
    elif 'No' in content and 'Yes' not in content:
        result.append('No')
    return content, result