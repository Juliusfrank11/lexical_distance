import wordcloseness as wc
import wordcloseness_constants as wc_con
import random as rd

# setting variables for data collection by year
sample_size = 30 # how many random words to pick out from 
languages = ('english','german','arabic','french','spanish','chinese','russian','hindi','indonesian','japanese','swahili','persian','polish','hungarian','turkish')
languages = [wc_con.inv_lang_dict[l] for l in languages] # convert to abbrivation
#what it says on the tin
sleep_time = 0.4 #time between translations

# data collection by year
for i in range(1593,1700):
    try:
        with open('wordbanks/' + str(i) + '.txt','r') as f:
            words = f.readlines()
        words = [w.strip() for w in words]
        if len(words) < sample_size: #not enough words to make sample? take what you can
            for word, n in zip(words,range(len(words))):
                if n == 0:
                    df = wc.gen_comp_matrix(word,languages,p=True,sleep=sleep_time)
                else:
                    df += wc.gen_comp_matrix(word,languages,p=True,sleep=sleep_time)
            df = df/len(words)
            df.to_csv(path_or_buf=str(i)+'.csv')
        else:
            sample = rd.choices(words,k=sample_size)
            for word, n in zip(sample,range(sample_size)):
                if n == 0:
                    df = wc.gen_comp_matrix(word,languages,p=True,sleep=sleep_time)
                else:
                    df += wc.gen_comp_matrix(word,languages,p=True,sleep=sleep_time)
            df = df/sample_size
            df.to_csv(path_or_buf=str(i)+'.csv')
    except FileNotFoundError:
        pass
    
    
