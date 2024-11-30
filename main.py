# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 22:52:41 2024

@author: anmol
"""

import pandas as pd
from model.predict_model import predict_model
match_no=int(input())
if match_no==1:
    df=pd.read_csv(r"C:\Users\anmol\prod_features\data\file_1.csv")
elif match_no==2:
    df=pd.read_csv(r"C:\Users\anmol\prod_features\data\file_2.csv")
else:
    df=pd.read_csv(r"C:\Users\anmol\prod_features\data\file_3.csv")
    
command=input()
player_id=input()
df_copy=df
df=df[df['player_id']==player_id]
(y_pred_sorted,dream_team_points)=predict_model(df_copy,'match_predictions','')
team1=y_pred_sorted[:11]
team2=y_pred_sorted[11:]
temp=dream_team_points

if command=='batting_strike_rate':
    from functionalities.features import batting_strike_rate
    strike_rate=batting_strike_rate(df)
elif command=='bowling_economy':
    from functionalities.features import bowling_economy
    economy=bowling_economy(df)
elif command=='pitch_score':
    from functionalities.features import pitch_score
    score=pitch_score(player_id,df['venue'].iloc[0])
elif command=='floor':
    from functionalities.features import floor
    floor_value=floor(player_id)
elif command=='ceil':
    from functionalities.features import ceil
    ceil_value=ceil(player_id)
elif command=='batting_first_fantasy_points':
    from functionalities.features import batting_first_fantasy_points
    (original_score,predicted_score)=batting_first_fantasy_points(player_id, df)
elif command=='chasing_first_fantasy_points':
    from functionalities.features import chasing_first_fantasy_points
    (original_score,predicted_score)=chasing_first_fantasy_points(player_id, df)
elif command=='relative_points':
    from functionalities.features import relative_points
    points=relative_points(df_copy, player_id)
elif command=='matchup_rank':
    from functionalities.features import matchup_rank
    rank=matchup_rank(df_copy,player_id)
elif command=='six_match_predictions':
    from functionalities.features import six_match_prediction
    (y_actual,y_pred,date_of_match)=six_match_prediction(df)
elif command=='transaction_rating':
    from functionalities.features import transaction_rating
    (team1,team2,temp)=transaction_rating(team1,team2,temp,player_id)
elif command=='AI_alert':
    from functionalities.ai import ai_alert
    text=ai_alert(player_id)
elif command=='team_spider_chart':
    from functionalities.features import team_spider_chart
    pfpp=dream_team_points/11
    (fes,doi,pcb)=team_spider_chart(df_copy,team1,dream_team_points)
    
