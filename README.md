# UGATIT

## Instructions

### Prepare the dataset
1. Download [selfie2anime](https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF/view?usp=sharing) (provided by the original implementation) 
1. Put the downloaded selfie2anime into data/ and unzip it there

### Kick Start Training
`$ python3 main.py --phase train`

### Test the Trained Model
`$ python3 main.py --phase test --ckpt <path_to_ckpt>`

## Directory Structure
```
-- UGATIT
   |- main.py
   |- model/
      |- test
         |- test_critic.py
         |- test_generator.py
         |- test_model.py
      |- __init__.py
      |- factory.py
      |- model.py
      |- critic.py
      |- generator.py
      |- modules.py
      |- helpers.py
   |- data_pipeline/
      |- test
         |- test.py
      |- __init__.py
      |- factory.py
      |- reader.py
      |- tf_helper.py
   |- helper
      |- __init__.py
      |- directory.py
   |- data
      |- selfie2anime
         |- trainA
            |- *.jpg
         |- trainB
            |- *.jpg
         |- testA
            |- *.jpg
         |- testB
            |- *.jpg
```
