# TODO 

### Improving training
- [] Create a program that will webscrape for more intents/patterns/responses
- [] Faster training with more threads

### Experimental
- [] More epochs of training once threaded
- [] allow bot to learn patterns while chatting and save in json
- [] Allow for decision tree, where if the bot doesn't know the response to a question or sentence, search it htrough google or something
     and respond back, learn from this and add to the json file or create another json file based on this session. Can you explain that to me kind of thing and it will write it to the json file the information it is given.

### Quality of life
- [] Flags for specifiying intents.json/training data file + data.pth/output training data file
- [] Checks for if data.pth exist or not and query if you would like to train if it doesn't already, take in json as input if so
