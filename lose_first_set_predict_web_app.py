import sys
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#from tensorflow import keras
from tensorflow.keras.models import load_model
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly
import numpy as np 


markdown_text1 = '''
The predictor uses a Machine Learning model, based upon many historical profressional matches, to estimate the probability of a player winning the match 
(assuming the match is 3 tie break sets) after losing the first set. 
*It attempts to uncover the latent state representing the ability of a player to make a comeback* and is based on 

1. The difference in the players ranking, which estimates their ability prior to the match \n
2. The difference in points won during the set \n
3. The number of games won by the losing player during the set, and \n
4. The number of breaks of serve achieved by the losing player during the set

Hence the model tries to combine the underlying ability, essentially the static ranking data available prior to the match, together with dynamic data available during the set to provide a 
representation from which a prediction can be made. The relationship between the predictors and the target is nearly linear hence two models are used - 
Logistic Regression (LR) and a Neural Network multilayer perceptron (MLP) with a single hidden layer to capture any non-linearity in the relationship. 
The model outputs both the probability and the inverse of the probability

'''
app = Dash()
server = app.server
model = load_model('weights.hdf5')
logreg_model = pickle.load(open('logreg_model.sav','rb'))
available_points_diff_indicators = list(np.linspace(-15,4,num=20))
available_games_for_set1_indicators = [0,1,2,3,4,5,6]
available_breaks_for_set1_indicators = [0,1,2,3,4]

app.layout = html.Div([html.Br(), html.Div([
             html.H1('Lose the First Set Tennis Predictor Model',style={'color': 'red', 'fontSize': 30,'textAlign': 'center'}),html.Br(),
             dcc.Markdown(markdown_text1),

        html.Div([
				  html.Span(html.Label('Firstly enter the players rankings'),style={'text-decoration':'underline','color': '#ff0000'}),html.Br(),html.Br(),
                  html.Label("\nPlayer Losing Set 1"),html.Div(dcc.Input(id='myrank',value='0', type='text')),html.Br(),html.Br(),
				  html.Span(html.Label('Now Enter the Relevant values from within the set and Check Out the Probabilities'),style={'text-decoration':'underline','color': '#ff0000'}),html.Br(),html.Br(),
		          html.Label('Points Diff for the set'),dcc.RadioItems(id='pointsdiff',options=[{'label': i, 'value': i} for i in available_points_diff_indicators],value=0),
				  html.Label('Games for the set'),dcc.RadioItems(id='games',options=[{'label': i, 'value': i} for i in available_games_for_set1_indicators],value=0),
				  html.Label('Breaks for the set'),dcc.RadioItems(id='breaks',options=[{'label': i, 'value': i} for i in available_breaks_for_set1_indicators],value=0)],
                  style={'width': '48%', 'display': 'inline-block'}),
				  
		html.Div([html.Br(),html.Br(),html.Label("\nPlayer Winning Set 1"),html.Div(dcc.Input(id='yourrank',value='0', type='text'))],
				  style={'width': '48%', 'float': 'right','display': 'inline-block'})
				  
]),

html.Div([
				html.H3(id='output-text1',style={'color': 'red','text-decoration':'underline'}),
				html.H3(id='output-text2',style={'color': 'red'}),
				html.H3(id='output-text3',style={'color': 'blue','whiteSpace': 'pre-wrap'}),
				html.H3(id='output-text4',style={'color': 'green','whiteSpace': 'pre-wrap'}),
				
])
])

@app.callback(
    [Output('output-text1', 'children'),
     Output('output-text2', 'children'),
     Output('output-text3', 'children'),
     Output('output-text4', 'children')],
    [Input('myrank', 'value'),
     Input('yourrank', 'value'),
     Input('pointsdiff', 'value'),
     Input('games', 'value'),
     Input('breaks', 'value')])

def update_score(player1_rank,player2_rank,points_diff_set1,games_for_set1,breaks_for_set1):
    ranking_diff_use = round(((int(player1_rank) - int(player2_rank))/5));
    ranking_diff_use = min(max(ranking_diff_use,-20),20);
    title = ''	
    title = title + 'Results for player1 with rank {} and player2 with rank {}\n'.format(player1_rank,player2_rank)    
    title1 = ''
    title1 = title1 + 'Points Diff is {}, games_for_set1 is {} and breaks_for_set1 is {}\n'.format(points_diff_set1,games_for_set1,breaks_for_set1)    
    points_diff_set1 = (points_diff_set1 + 15)/19
    games_for_set1 = (games_for_set1 + 0)/6
    breaks_for_set1 = (breaks_for_set1 + 0)/4
    ranking_diff_use = (ranking_diff_use + 20)/40
    tmp = model.predict(np.array([ranking_diff_use,points_diff_set1,games_for_set1,breaks_for_set1]).reshape(1,-1))
    tmp1 = logreg_model.predict_proba(np.array([ranking_diff_use,points_diff_set1,games_for_set1,breaks_for_set1]).reshape(1, -1))[0,1]
    result1 = '\nProbability for the Player\n' 
    result1 = result1 + f'MLP {float(tmp[0]):0.4f}' + ' ' + f'({1/float(tmp[0]):0.4f})' + f'  LR {float(tmp1):0.4f}' + ' ' + f'({1/float(tmp1):0.4f})'
    result2 = '\nProbability for the Opposition\n' 
    result2 = result2 + f'MLP {(1-float(tmp[0])):0.4f}' + ' ' + f'({1/(1-float(tmp[0])):0.4f})' + f'  LR {(1-float(tmp1)):0.4f}' + ' ' + f'({1/(1-float(tmp1)):0.4f})'   
    return title,title1,result1,result2


if __name__ == '__main__':
    app.run_server()
