# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests

model_inputs = {'fileName': 'Charles C. Mann - The Wizard and the Prophet_ Two Remarkable Scientists and Their Dueling Visions to Shape Tomorrow’s World-Knopf Publishing Group (2018).epub', 'downloadURL': 'https://firebasestorage.googleapis.com/v0/b/book-search-367318.appspot.com/o/books%2FCharles%20C.%20Mann%20-%20The%20Wizard%20and%20the%20Prophet_%20Two%20Remarkable%20Scientists%20and%20Their%20Dueling%20Visions%20to%20Shape%20Tomorrow%E2%80%99s%20World-Knopf%20Publishing%20Group%20(2018).epub?alt=media&token=46fcf428-e583-48cd-b8cd-6d75b65ffcc2', 'query': 'impact of climate change'}

res = requests.post('http://localhost:8000/', json = model_inputs)

print(res.json())