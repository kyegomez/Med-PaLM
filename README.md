[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


# Med-PaLM 
A responsible path to generative AI in healthcare: Unleash the power of Med-PaLM 2 to revolutionize medical knowledge, answer complex questions, and enhance healthcare experiences with accuracy, safety, and equitable practices.

[![GitHub issues](https://img.shields.io/github/issues/kyegomez/Med-Palm)](https://github.com/kyegomez/Med-Palm/issues) 
[![GitHub forks](https://img.shields.io/github/forks/kyegomez/Med-Palm)](https://github.com/kyegomez/Med-Palm/network) 
[![GitHub stars](https://img.shields.io/github/stars/kyegomez/Med-Palm)](https://github.com/kyegomez/Med-Palm/stargazers) [![GitHub license](https://img.shields.io/github/license/kyegomez/Med-Palm)](https://github.com/kyegomez/Med-Palm/blob/master/LICENSE)
[![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/Med-Palm)](https://twitter.com/intent/tweet?text=Excited%20to%20introduce%20Med-Palm,%20the%20all-new%20robotics%20model%20with%20the%20potential%20to%20revolutionize%20automation.%20Join%20us%20on%20this%20journey%20towards%20a%20smarter%20future.%20%23RT1%20%23Robotics&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMed-Palm)
[![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMed-Palm)
[![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMed-Palm&title=Introducing%20Med-Palm%2C%20the%20All-New%20Robotics%20Model&summary=Med-Palm%20is%20the%20next-generation%20robotics%20model%20that%20promises%20to%20transform%20industries%20with%20its%20intelligence%20and%20efficiency.%20Join%20us%20to%20be%20a%20part%20of%20this%20revolutionary%20journey%20%23RT1%20%23Robotics&source=)
![Discord](https://img.shields.io/discord/999382051935506503)
[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMed-Palm&title=Exciting%20Times%20Ahead%20with%20Med-Palm%2C%20the%20All-New%20Robotics%20Model%20%23RT1%20%23Robotics) [![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMed-Palm&t=Exciting%20Times%20Ahead%20with%20Med-Palm%2C%20the%20All-New%20Robotics%20Model%20%23RT1%20%23Robotics)
[![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMed-Palm&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=Med-Palm%2C%20the%20Revolutionary%20Robotics%20Model%20that%20will%20Change%20the%20Way%20We%20Work%20%23RT1%20%23Robotics)
[![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=I%20just%20discovered%20Med-Palm,%20the%20all-new%20robotics%20model%20that%20promises%20to%20revolutionize%20automation.%20Join%20me%20on%20this%20exciting%20journey%20towards%20a%20smarter%20future.%20%23RT1%20%23Robotics%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2FMed-Palm)


![Med palm](image-9.png)

## Getting Started

```
pip install MedPalm
```

## Usage


```python
import torch
from medpalm.model import MedPalm

#usage
img = torch.randn(1, 3, 256, 256)
caption_tokens = torch.randint(0, 4)

model = MedPalm()
output = model(img, caption_tokens)
```
📝 Note: Modify the examples to suit your data and project needs.

## 📚 Datasets 
- Wanna deep-dive? [Click here for a dive into dataset strategies](docs/DATASETS.md)

## 💼 Commercial Use-Cases

Med-PaLM isn't just fun, it's super useful! 🛍️
- **Clinical Diagnostics**: Combining medical imaging, patient tales 📖, and genes, we're aiming for top-notch diagnostic solutions.
  
- **Healthcare Research**: Dive deep into diverse datasets and discover something new with Med-PaLM by your side! 🤿
  
- **Telemedicine**: Quick, reliable, and remote! 🌍 Med-PaLM's here to revolutionize telehealth.

# Contributing to Med Palm 🤖🌟

First off, big high fives 🙌 and thank you for considering a contribution to Med Palm! Your help and enthusiasm can truly elevate this project. Whether you're fixing bugs 🐛, adding features 🎁, or just providing feedback, every bit matters! Here's a step-by-step guide to make your contribution journey smooth:

## 1. Set the Stage 🎬

**Fork the Repository:** Before you dive in, create a fork of the Med Palm repository. This gives you your own workspace where you can make changes without affecting the main project.

1. Go to the top right corner of the Med Palm repo.
2. Click on the "Fork" button. 

Boom! You now have a copy on your GitHub account.

## 2. Clone & Set Up 🚀

**Clone Your Fork:** 
```bash
git clone https://github.com/kyegomez/Med-PaLM.git
cd Med-PaLM
```

**Connect with the Main Repo:** To fetch updates from the main Med Palm repository, set it up as a remote:
```bash
git remote add upstream https://github.com/kyegomez/Med-PaLM.git
```

## 3. Make Your Magic ✨

Create a new branch for your feature, bugfix, or whatever you're looking to contribute:
```bash
git checkout -b feature/my-awesome-feature
```

Now, dive into the code and sprinkle your magic!

## 4. Stay Updated 🔄

While you're working, the main Med Palm repository might have updates. Keep your local copy in sync:

```bash
git fetch upstream
git merge upstream/main
```

## 5. Share Your Brilliance 🎁

Once you've made your changes:

1. **Stage & Commit:**
   ```bash
   git add .
   git commit -m "Add my awesome feature"
   ```

2. **Push to Your Fork:**
   ```bash
   git push origin feature/my-awesome-feature
   ```

3. **Create a Pull Request:** Head back to your fork on GitHub, and you'll see a "New Pull Request" button. Click on it!

## 6. The Review Dance 💃🕺

Once your PR is submitted, our team will review it. They might have questions or feedback. Stay engaged, discuss, and make any needed changes. Collaboration is key! 🤝

## 7. Celebrate 🎉

After review and any necessary tweaks, your contribution will be merged. Pat yourself on the back and celebrate! 🎊

## 8. Spread the Word 📢

Share about your contribution with your network. The more the merrier! Plus, it feels good to show off a bit, right? 😉

Remember, every contribution, no matter how small or large, is valued and appreciated. It's the collective effort that makes open-source so vibrant and impactful. Thanks for being a part of the Med Palm adventure! 🌟🚀

----

## License

Med-PaLM's is under the MIT license. Check out the details [here](LICENSE.md).

## Citation

Tao Tu, Shekoofeh Azizi, Danny Driess, Mike Schaekermann, Mohamed Amin, Pi-Chuan Chang, Andrew Carroll, Chuck Lau, Ryutaro Tanno, Ira Ktena, Basil Mustafa, Aakanksha Chowdhery, Yun Liu, Simon Kornblith, David Fleet, Philip Mansfield, Sushant Prakash, Renee Wong, Sunny Virmani, Christopher Semturs, S Sara Mahdavi, Bradley Green, Ewa Dominowska, Blaise Aguera y Arcas, Joelle Barral, Dale Webster, Greg S. Corrado, Yossi Matias, Karan Singhal, Pete Florence, Alan Karthikesalingam, Vivek Natarajan. "Towards Generalist Biomedical AI." arXiv:2307.14334 [cs.CL], July 26, 2023.
