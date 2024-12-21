import requests
import pickle
from urllib.parse import quote
import os
import random
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset, Audio
import torch
import wave
import io

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

sentences = [
    "She hummed softly while painting a sunset on the canvas.",
    "The forest whispered secrets in the rustling of its leaves.",
    "The wind howled through the narrow canyon, carrying dust and whispers.",
    "Time seemed to slow as he reached out to catch the falling glass.",
    "The market bustled with life as vendors shouted and customers bargained.",
    "Rain poured in torrents, drenching everything under the steel-gray sky.",
    "A butterfly landed on her outstretched hand.",
    "The old man sat quietly on the park bench, watching the world go by. its wings delicate as lace",
    "The ship sailed into the harbor, its sails billowing in the wind.",
    "She stood in the doorway, clutching a letter that changed everything.",
    "The lake reflected the sky, its surface as smooth as glass.",
    "The train sped through the countryside.",
    "The clock ticked loudly in the silent room, marking each passing second. It fields blurring into patches of green",
    "Under the moonlit sky, they danced to a song only they could hear.",
    "The detective scanned the scene, piecing together the fragments of the mystery.",
    "The child stared wide-eyed at the magician pulling rabbits from his hat.",
    "The dog barked happily, wagging its tail at the sound of its owner’s voice.",
    "The thunderstorm rolled in suddenly, darkening the sky and drenching the streets.",
    "A small bird chirped on the windowsill.",
    "The baby giggled, her laughter filling the room with pure joy.",
    "She found an old photograph in the box",
    "The bridge swayed slightly in the wind as they crossed it nervously.",
    "He opened the dusty book, its pages crackling as they turned. Its edges frayed with time.",
    "The sunset painted the horizon with hues of orange, pink, and purple. The bird breaking the stillness of the morning.",
    "The little girl built a sandcastle, decorating it with shells and seaweed.",
    "The forest came alive with the chirping of crickets as night fell.",
    "The painter mixed vibrant colors on her palette.",
    "The sound of waves crashing against the rocks was oddly soothing. Her brush poised midair.",
    "The city skyline sparkled like a jewel under the clear night sky.",
    "He spent the entire afternoon fixing the squeaky hinge on the door.",
    "The aroma of fresh coffee filled the air.",
    "The boy laughed as his kite soared high above the grassy field.",
    "A mysterious package arrived at the doorstep, addressed to someone unknown.",
    "The wind carried the scent of blooming flowers from the meadow below.",
    "The eagle soared above the valley, its wings spread wide against the sky. It promising a much-needed pick-me-up.",
    "The cat stretched luxuriously on the windowsill.",
    "The classroom buzzed with excitement as the teacher announced a surprise activity. The cat basking in the warmth of the sun",
    "The storm subsided, leaving puddles that mirrored the blue of the sky.",
    "The fire crackled in the hearth, casting warm light across the cozy room.",
    "The hikers stopped to admire the view, the mountains stretching endlessly before them.",
    "The garden bloomed with vibrant flowers, each petal a splash of color.",
    "The librarian smiled kindly as she handed the little boy his first library card.",
    "The couple shared a quiet moment, watching the stars in the tranquil night.",
    "She hesitated before knocking on the door.",
    "The crowd roared with excitement as the team scored the winning goal.",
    "The butterfly flitted from flower to flower, its movements graceful and deliberate.",
    "The snow fell gently, blanketing the world in a soft layer of white.",
    "The artist studied her reflection in the mirror, searching for inspiration. She unsures of what to expect",
    "The fox darted into the underbrush, its tail disappearing into the shadows.",
    "The baby held her father’s finger tightly, her tiny hand full of trust.",
    "The mountains loomed in the distance, their peaks wrapped in wisps of clouds.",
    "She opened the door to a room filled with balloons and happy faces.",
    "The scientist carefully adjusted the microscope, peering intently at the specimen.",
    "The ocean stretched endlessly before them.",
    "The fisherman cast his net, hoping for a good catch before the sun set. The horizon is a thin line of light.",
    "The cake was decorated with delicate flowers, each one crafted from sugar.",
    "The boy planted a seed, watering it daily and waiting for it to sprout.",
    "The house stood abandoned, its windows broken and walls covered in ivy.",
    "The baby reached for the colorful toy.",
    "The farmer worked tirelessly in his fields, his hands rough and sun-worn.",
    "The parade wound through the streets, a colorful spectacle of music and dance.",
    "She closed her eyes and listened to the rain, feeling at peace. The baby giggling toy as it made a soft sound.",
    "The squirrel scampered up the tree, clutching an acorn in its tiny paws.",
    "The photographer waited patiently for the perfect moment to capture the sunrise.",
    "The sun rose slowly, bathing the mountains in a warm golden glow.",
    "The boy ran through the field, his laughter carried on the wind.",
    "The market smelled of fresh spices, ripe fruits, and freshly baked bread.",
    "The astronaut floated weightlessly, marveling at the vastness of space around her.",
    "The letter arrived in a yellow envelope.",
    "The toddler stumbled across the room, determined to reach her favorite toy. The handwriting unfamiliar but intriguing.",
    "The couple danced under the stars, their laughter echoing in the cool night air.",
    "The detective found a crucial clue hidden in plain sight at the crime scene.",
    "The train whistle echoed through the valley, announcing its arrival at the station.",
    "The musician strummed his guitar, filling the room with a soulful melody.",
    "The snowman melted slowly in the afternoon sun.",
    "The treehouse creaked as the children climbed into their secret hideaway.",
    "The bird flew above the trees, its silhouette sharp against the orange sky. The rain leaving only a puddle behind.",
    "The rain pattered softly on the rooftop, creating a soothing rhythm that lulled her to sleep.",
    "The old clock chimed midnight, its sound echoing through the empty halls.",
    "The waves rolled onto the shore, each one erasing the footprints left behind.",
    "The girl spun around in her new dress, feeling like a princess for a day.",
    "The wind tugged at the kite, pulling it higher into the clear blue sky.",
    "The city came alive as night fell.",
    "The adventurer stood at the summit, breathing in the crisp mountain air.",
    "The candle flickered in the dark room, casting dancing shadows on the walls.",
    "The children played tag in the park, their laughter ringing through the air.",
    "The fireflies lit up the garden like tiny floating lanterns.",
    "The bakery's display was filled with pastries, each one more tempting than the last. The streets glowing with vibrant neon lights",
    "The baby fell asleep in her mother’s arms, her tiny fingers clutching her shirt.",
    "The artist stared at the blank canvas, feeling both excitement and trepidation.",
    "The storm gathered strength, clouds turning an ominous gray as the wind picked up",
    "The astronaut floated in the spacecraft, gazing out at the endless expanse of stars and galaxies. She stood on the edge of the cliff.",
    "The child carefully placed her tooth under the pillow.",
    "The hiker paused to admire the view from the summit. The world seemed to stretch endlessly before him.",
    "The mysterious package arrived on her doorstep. She opened it cautiously, unsure of its contents.",
    "The stars twinkled. They are so beautiful.",
    "The young couple danced under the fairy lights, their laughter echoing through the cool evening air.",
    "He brushed the dust off the old photo album and opened it.",
    "The dog chased its tail in endless circles, much to the amusement of everyone at the park.",
    "The sound of rain on the roof was soothing, lulling her into a deep and restful sleep."
]

def generate_audio():
    # Coqui TTS API endpoint
    COQUI_URL = "http://141.223.124.62:30502/api/tts"  # Replace with actual server IP and port

    # sentences = sentences[:1]


    # Dictionary to store generated audio data
    audio_data_arr = []
    audio_len_arr = []

    # Loop through each word and send it to the Coqui TTS API
    # mid_idx = len(sentences)//2
    # for i in range(mid_idx):
    for sentence in sentences:
        # if i < 20:
        #     sentence = sentences[i]
        # else:
        #     sentence = f'{sentences[i]} {sentences[i+mid_idx]}'
        for speaker_id in ['p226', 'p227', 'p228', 'p229', 'p230', 'p231', 'p232', 'p233', 'p234', 'p236']:
        # for speaker_id in ['p376']:
            # Payload for the Coqui TTS API
            params = {
                "text": sentence,
                "speaker_id": speaker_id
            }

            # Send request to the Coqui TTS endpoint
            response = requests.post(COQUI_URL, params=params)

            if response.status_code == 200:
                # Save the audio data as raw binary
                audio_sample = response.content
                audio_data_arr.append(audio_sample)

                audio_buffer = io.BytesIO(response.content)
                with wave.open(audio_buffer, "r") as audio:
                    frames = audio.getnframes()
                    rate = audio.getframerate()
                    duration = frames / float(rate)
                    audio_len_arr.append(duration)

                print(f"Generated audio for: {sentence}, len: {duration}")
                # with open(f'{BASE_DIR}/test.wav', "wb") as f:
                #     f.write(response.content)
                #     break
            else:
                print(f"Failed to generate audio for: {sentence}")
                print(f"Error: {response.text}")

    print(f"summary: min: {min(audio_len_arr)}, max: {max(audio_len_arr)}, avg: {sum(audio_len_arr)/len(audio_len_arr)}")
    # summary: min: 2.0440816326530613, max: 11.041814058956916, avg: 4.1923874829932
    return audio_data_arr

def generate_questions(num_questions=500):
    questions = []

    # Math templates (reduced focus)
    math_templates = [
        "What is {} + {}?",
        "What is {} - {}?",
        "What is {} x {}?"
    ]

    # Predefined data
    countries = [
        "France", "Italy", "Germany", "Japan", "India", "Brazil",
        "Canada", "Australia", "Russia", "Egypt", "China", "United States",
        "Mexico", "South Africa", "United Kingdom", "Argentina", "Spain",
        "South Korea", "Indonesia", "Turkey"
    ]
    things = [
        "color", "season", "continent", "planet", "shape", "animal",
        "vegetable", "fruit", "flower", "bird", "insect", "tree",
        "metal", "gemstone", "sport", "tool", "country", "language",
        "instrument", "vehicle", "book", "movie", "river", "ocean",
        "mountain", "building", "city", "food", "drink", "star"
    ]
    opposites = [ ("hot", "cold"), ("up", "down"), ("big", "small"), ("fast", "slow"), ("light", "dark"), ("hard", "soft"),
        ("early", "late"), ("strong", "weak"), ("happy", "sad"), ("full", "empty"), ("near", "far"), ("old", "young"), ("rich", "poor"),
        ("clean", "dirty"), ("short", "tall"), ("thin", "thick"), ("heavy", "light"), ("left", "right"), ("right", "wrong"), ("open", "closed")
    ]
    actions = ["fly", "swim", "run", "jump", "climb"]
    famous_discoveries = [ "gravity", "penicillin", "electricity", "radio waves", "the theory of relativity", "DNA structure", "germ theory",
        "atomic structure", "heliocentric model", "calculus", "vaccination", "periodic table", "radioactivity", "x-rays", "quantum mechanics",
        "antibiotics", "photosynthesis", "evolution by natural selection", "the big bang theory", "plate tectonics"
    ]
    alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    sports = ["soccer", "cricket", "basketball", "tennis", "hockey", "baseball"]
    leaders = ["president", "prime minister", "monarch", "chancellor"]

    # Generate math questions (10%)
    numbers = list(range(1, 21))  # Numbers for math
    for a in numbers[:10]:
        for b in numbers[10:]:
            template = math_templates[(a + b) % len(math_templates)]
            questions.append(template.format(a, b))
            if len(questions) >= num_questions // 10:
                break
        if len(questions) >= num_questions // 10:
            break

    # Generate diverse general knowledge questions
    for country in countries:
        questions.append(f"What is the capital of {country}?")
        questions.append(f"What is the official language of {country}?")
        questions.append(f"What is the national dish of {country}?")
        questions.append(f"What is the currency of {country}?")
        questions.append(f"What is the nickname of {country}?")
        questions.append(f"What is the national flower of {country}?")
        questions.append(f"What sport is {country} famous for?")
        questions.append(f"What is the flag color of {country}?")
        questions.append(f"Who is the {random.choice(leaders)} of {country}?")
        questions.append(f"What is the GDP of {country}?")

    for thing in things:
        questions.append(f"Name a {thing}.")
        questions.append(f"What is the largest {thing}?")
        questions.append(f"What is the smallest {thing}?")
        questions.append(f"Name a famous {thing}.")

    for opposite in opposites:
        questions.append(f"What is the opposite of {opposite[0]}?")

    for action in actions:
        questions.append(f"Name an animal that can {action}.")

    for discovery in famous_discoveries:
        questions.append(f"Who discovered {discovery}?")

    for letter in alphabets:
        questions.append(f"Name a country that starts with {letter}.")

    for planet in planets:
        questions.append(f"Which planet is {planet}?")

    for sport in sports:
        questions.append(f"Which country is known for {sport}?")

    # Dynamic questions (e.g., square, greater/less than)
    for number in numbers:
        questions.append(f"Name a number less than {number}.")
        questions.append(f"Name a number greater than {number}.")
        questions.append(f"What is the square of {number}?")
        questions.append(f"What is the square root of {number}?")

    # Shuffle and limit to num_questions
    questions = questions[:num_questions]
    random.shuffle(questions)
    return questions

from datasets import load_dataset, Audio
import torchaudio.transforms as T

# Function to resample and convert to mono
def resample_to_16khz_and_mono(audio_sample):
    # Extract waveform and sample rate
    waveform = audio_sample["audio"]["array"]
    sample_rate = audio_sample["audio"]["sampling_rate"]

    # Convert waveform to torch tensor
    waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

    # If stereo, convert to mono by averaging channels
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
    resampled_waveform = resampler(waveform)

    # Update the audio_sample with resampled waveform and new sample rate
    audio_sample["audio"]["array"] = resampled_waveform.squeeze().numpy()
    audio_sample["audio"]["sampling_rate"] = 16000
    return audio_sample



# # Access a resampled audio sample
# resampled_sample = resampled_dataset[0]["audio"]
# print(f"Shape of resampled audio: {resampled_sample['array'].shape}")
# print(f"Sampling rate: {resampled_sample['sampling_rate']}")

# # Save the first resampled audio to a WAV file (optional)
# torchaudio.save("resampled_output.wav", torch.tensor(resampled_sample["array"]).unsqueeze(0), 16000)

def load_mozzila_common_voice():
    dataset = load_dataset("mozilla-foundation/common_voice_11_0", 'en', split='test', trust_remote_code=True)
    dataset = dataset.select(range(2000))
    dataset = dataset.filter(lambda x: len(x["audio"]["array"])/x['audio']['sampling_rate'] <= 10.0 and len(x["audio"]["array"])/x['audio']['sampling_rate'] > 3.0)
    dataset = dataset.select(range(500))
    # dataset = dataset.cast_column("audio", Audio())

    # Apply the transformation to the dataset
    dataset = dataset.map(resample_to_16khz_and_mono)
    audio_samples = []
    for sample in dataset:
        audio_samples.append(sample["audio"]["array"].tobytes())
    return audio_samples

def get_image_dataset_triton():
    # Load CIFAR-10 Dataset
    image_dataset = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 for ResNet input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # Download CIFAR-10 dataset
    cifar10_dataset = torchvision.datasets.CIFAR10(
        root='~./drafas',
        train=False,
        download=True,
        transform=transform
    )

    for i in random.sample(range(len(cifar10_dataset)), 500):
        image_data, _ = cifar10_dataset[i]
        image_data_converted = image_data.numpy().flatten().tolist()
        image_dataset.append(image_data_converted)

    return image_dataset


def get_image_dataset_pytorch():
    # Load CIFAR-10 Dataset
    image_dataset = []
    transform = transforms.Compose([
        transforms.Resize(512),
        # transforms.CenterCrop(224),
        # transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # ),
        # transforms.ToPILImage(mode='RGB')
    ])

    # Download CIFAR-10 dataset
    cifar10_dataset = torchvision.datasets.CIFAR10(
        root='~./drafas',
        train=False,
        download=True
    )

    for i in random.sample(range(len(cifar10_dataset)), 500):
        image, _ = cifar10_dataset[i]
        image = transform(image)
        # image = image.numpy()
        # image = transforms.ToTensor()
        image_dataset.append(image)

    return image_dataset



if __name__ == '__main__':
    # audio_data_arr = load_mozzila_common_voice()
    # audio_data_arr = generate_audio()
    # with open(f"{BASE_DIR}/data/audio_data.pkl", "wb") as f:
    #     pickle.dump(audio_data_arr, f)

    # llm_questions = generate_questions(500)
    # with open(f"{BASE_DIR}/data/llm_questions_data.pkl", "wb") as f:
    #     pickle.dump(llm_questions, f)

    # image_dataset = get_image_dataset_triton()
    # with open(f"{BASE_DIR}/data/image_dataset.pkl", "wb") as f:
    #     pickle.dump(image_dataset, f)

    image_dataset = get_image_dataset_pytorch()
    with open(f"{BASE_DIR}/data/image_dataset_pytorch.pkl", "wb") as f:
        pickle.dump(image_dataset, f)
