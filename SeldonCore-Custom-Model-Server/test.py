# Import required dependencies
import requests
import json
import json
from mlserver.types import InferenceResponse
from mlserver.codecs.string import StringRequestCodec

inputs = {
  "tweets": [
    "Today I learned that Leonardo DiCaprio dating a 19-year-old adult woman is the height of moral evil, mostly from the same people who believe a 4-year-old can consent to gender change.",
    "In which @mattwalshblog leaves an elected Democrat absolutely speechless with one simple question. This clip is stunning.  https://t.co/R9hsy9hSlb",
    "Check out my full reaction to the State of the Union address here: https://t.co/YheEJblr3F https://t.co/Mln0zPbPce",
    "This is the message of our modern-day Satanists as well. Religion and morality are evils. Worship of “authenticity,” particularly in the sexual realm, is the highest possible good. Worship Satan by worshipping yourself.\n\nBrought to you by Pfizer and Dr. Jill Biden, gang!",
    "Which problems, specifically? Do you mean massive spikes in teen suicidal ideation, forty percent of babies born out of wedlock, social contagions around nonsense gender and sexual identities? That ain't parental sex ed.  https://t.co/WXVtKdz6sh",
    "I have said on air that if I knew then what I know now, I wouldn't have taken it. But no matter my personal feelings about the vaxx and the evolution of the info we were given, I NEVER supported mandates. I risked my business to oppose them. Name another company that did.",
    "I recommended the vaxx to the elderly and the obese to reduce death rates. I recommended the vaccine for people my age because we were lied to about reduction in transmissibility.  That's why I took it personally. I never recommended it for the young. My kids are not vaxxed.",
    "We sued the federal government and refused to force our employees to vaxx or test. I noticed you have said nothing about Fox News' covid policy, which mandated that its employees vaxx or test every day. Fascinating.",
    "Hunter Biden Is An Evil Scumbag, Not A Victim | Ep. 1659  https://t.co/SkSZWZgvOz",
    "You shut down schools, killed small businesses en masse, forcibly masked our kids, tried to force businesses to vaxx their employees, banned us from travel, kept us away from our dying relatives...all based on science you KNEW was shoddy. F*** you.   https://t.co/GaHVN3xu2r"
  ]
}

inputs_string = json.dumps(inputs)

inference_request = {
    "inputs": [
        {
          "name": "echo_request",
          "shape": [len(inputs_string)],
          "datatype": "BYTES",
          "data": [inputs_string],
        }
    ]
}

inference_url = "http://localhost:8080/v2/models/text-uniqueness-model/infer"

response = requests.post(inference_url, json=inference_request)
print(f"full response:\n")
print(response)
# retrive text output as dictionary
inference_response = InferenceResponse.parse_raw(response.text)
raw_json = StringRequestCodec.decode_response(inference_response)
output = json.loads(raw_json[0])
print(f"\ndata part:\n")
print(output)