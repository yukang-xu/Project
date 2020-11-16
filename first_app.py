#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import pandas as pd
import streamlit as st
st.title('Meet your NBA Assistant Manager:') 
st.header('Recommender System and next best transaction')
st.text("")
st.text("")
st.write('Yukang Xu')



st.text("")
st.text("")
st.text("")



st.header('1. Introduction')
st.write('My inspiration is coming from one of my favorite movie, Moneyball, in which a small-market team leveraged more analytical gauges of player performance successfully to field a competitive team.')

from PIL import Image
image = Image.open('moneyball-movie.jpg')
st.image(image,use_column_width=True)


st.write('My mission in this project is to implement that idea and turn it into an automated recommender system.')

st.text("")
st.text("")
st.text("")
st.header('2. Stability is the key to win')
st.subheader('Injury and transaction are the key to ruin stability of the team which is the key to succeed.')
st.subheader('a. Injury')
st.write('Since 2017, the average amount of games missed due to injury in top 5 team is apparently lower than the rest.')
from PIL import Image
image = Image.open('Picture2.png')
st.image(image, caption='Games Missed per player due to injury: Top5 team VS The rest ',use_column_width=True)

st.text("")
st.subheader('b. Transaction')
st.write('Chart below shows a positive relationship between standing and the number of trades.')
from PIL import Image
image = Image.open('Picture3.png')
st.image(image, caption='AVG number of Transaction since 2017 ',use_column_width=True)


st.text("")
st.text("")
st.text("")

st.header('3. How to use?')
st.write('We use recommender system to help manager find suitable players in the market when players got injured or transaction is unavoidable')
st.subheader('a. Find suitable one to replace injured players')
st.write('if plugging in a name, this system could narrow down our trading targets similar as key players who will be leaving and we could expect they play the almost the same role in the team. ')
st.selectbox('player name', ['Yao Ming'] + ['Kobe Bryant'])

df = pd.DataFrame(np.random.randn(5, 8),columns=('James Harden','Stephen Curry','Michael Jordan','Klay Tompson','Russ Westbrook','Chris Paul','Danny Green','Paul George'),index=('Score','Assistant','Rebound','Height','Weight'))
st.dataframe(df)  # Same as st.write(df)

st.text("")

st.subheader('b. List transfer targets ')
st.write('Once you type in a team’s name such as LA Lakers, it assists our manger to analyze current trading market and list the best fits to the team. They could be the perfect substitute players without losing chemistry of the team. ')
st.selectbox('Team name', ['La Lakers'] + ['Houston Rocket'])
df = pd.DataFrame(np.random.randn(5, 8),columns=('James Harden','Stephen Curry','Michael Jordan','Klay Tompson','Russ Westbrook','Chris Paul','Danny Green','Paul George'),index=('Score','Assistant','Rebound','Height','Weight'))
st.dataframe(df)  # Same as st.write(df)

st.text("")
st.text("")
st.write('Find me at Linkedin: www.linkedin.com/in/yukangxu')
