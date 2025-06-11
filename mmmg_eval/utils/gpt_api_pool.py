gpt_api_pool = [
#TODO: Add your GPT API pool configuration here
]

'''
AzureOpenAI Resource example:

{
    "index": 13,
    "azure_endpoint": YOUR_ENDPOINT,
    "api_key":YOUR_API_KEY,
    "api_version": PREVIEW,
    "rate_limit_requests": RATE_LIMIT
    "rate_limit_tokens": TOKEN_NUMS,
    "deployment_name": "o3"
}

NOTICE PLEASE DISABLE THE AZURE CONTENT FILTER.

We recommend to use the "o3" deployment from official OpenAI website, which is the most powerful model available.
'''