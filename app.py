# -*- coding: utf-8 -*-
"""
Created: Sep 14, 2018
@author: Aveedibya Dey
Email: aveedibya@gmail.com
---
Regression Simulator:
    - Simulate different regression models in Python and visualize the outputs
    - Regression models supported: OLS, SVR, RANSAC and TheilSen
    - These models are used as available in sklearn package
    - Dash module is used to generate the user ineraction tools
    - Splitting data into train:test is not available currently to help visualize data
    
"""
#Pandas, Numpy and Sklearn modules
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn import linear_model

#Dash and Plotly modules
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go

#Custom modules
from parse_contents import parse_contents
from sample_datasets import sample_data

app = dash.Dash()
app.title = 'Regression Simulator'

server = app.server

subheading_markdown = '''
__Quick Description__: 
Use the simulator below to try different regression models on your data. 
'''
subtext = '''
Currently supports:
 - Ordinary Least Squares Regression or OLS Linear Regression
 - Support Vector Regression with Radial-Basis, Linear, Polynomial and Sigmoid Kernel Functions
 - Robust Linear Regression
 - Theilsen Regression
'''

footnote_markdown = '''
Created by: Aveedibya Dey | [Source Code](https://github.com/aveedibya/regressionSimulator)
    '''

app.layout = html.Div(children=[
        
        #---------
        #Heading
        dcc.Markdown(children='''#### Regression Model Simulator '''),
        
        #---------
        #Sub-heading
        html.Div([dcc.Markdown(subheading_markdown)], 
                  style={'borderBottom': 'thin lightgrey solid', 'padding': '2'}),
                
        #---------
        #Chart
        html.Div(dcc.Graph(id='plot-data'), 
                 style={'padding': '0'}),
                 
        #--------
        html.Hr(),
        
        #---------
        #Controls
        html.Div([html.Div([html.Label('Select Regression Model:'),
                           dcc.Dropdown(id='model-dropdown',
                               options=[
                            {'label': 'Simple Linear Regression', 'value': 1},
                            {'label': 'Support Vector Regression', 'value': 2},
                            {'label': 'Robust Linear Regression', 'value': 3},
                            {'label': 'Theilsen Linear Regression', 'value': 4}],
                            placeholder="Select Regression Model", multi=True)]),
                  #------------
                  #SVR Options
                  html.Div(id='svr-inputs', children=[
                          html.Label('Select SVR Kernel Function:'),
                          dcc.Dropdown(id='kernel-selector',
                                options=[
                                    {'label': 'Radial Basis Function', 'value': 'rbf'},
                                    {'label': 'Linear', 'value': 'linear'},
                                    {'label': 'Polynomial', 'value': 'poly'},
                                    {'label': 'Sigmoid', 'value': 'sigmoid'}
                                    ], placeholder="Select SVR Kernel", multi=True),
                         html.Div([
                                 html.Label('Cost Function Coefficient:'),
                                 dcc.Input(id='cost-function', type='number', step=0.01, placeholder='C=1.0'),
                                 html.Label('Epsilon for Support-Vector:'),
                                 dcc.Input(id='epsilon', type='number', step=0.01, placeholder='epsilon=0.5')]),
                        dcc.Checklist(id='hide-svr-tolerance', options=[{'label': 'Show SVR Tolerance', 'value':1}], values=[1])
                        ], 
                style={'background-color': '', 'padding': '20', 'display': 'none'})
        ], style={'display': 'inline-block', 'vertical-align': 'top', 'padding': '10', 'width': '45%'}),

        #--------
        #Block Below Graph
        html.Div([html.Label('Select Files and Regression Columns:'),
        
        #--------
        #Column Selection
        html.Div(id='column-options', children=[html.Label('Independent Variable, X:'),
                  dcc.Dropdown(id='X-dropdown'),
                  html.Label('Dependent Variable, y:'),
                  dcc.Dropdown(id='y-dropdown')
                  ]),
        #--------
        #Data Selection Elements
        html.Div([dcc.Upload(id='upload-data', 
                children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                style={
                    'width': '60%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '0px',
                    'borderStyle': 'solid','borderRadius': '5px', 'textAlign': 'center', 'margin-top': '5px', 
                    'backgroundColor': '#ebebe0'},
            ),
            html.Div(id='output-data-upload'),
            html.Div([html.Label(dcc.Markdown('*__OR__ Select Built-in Sample Data:*')),
                      html.Div(dcc.Dropdown(id='select-dataset',
                           options=[
                               {'label': 'No Outliers', 'value': 'df_normal'},
                               {'label': 'Small Outliers in x', 'value': 'df_x_errors'},
                               {'label': 'Small Outliers in y', 'value': 'df_y_errors'},
                               {'label': 'Large Outliers in x', 'value': 'df_x_large_errors'},
                               {'label': 'Large Outliers in y', 'value': 'df_y_large_errors'},
                               {'label': 'Sine Function with Noises', 'value': 'df_rbf_eps'},
                               {'label': 'Sine Function with Outliers', 'value': 'df_rbf_c'}
                           ],
                           placeholder='Built-in Dataset Not Selected',
                           value=''
                           ), style={})], style={'backgroundColor': '#ebebe0', 
                                                 'margin-top': '5px', 'padding': '10', 'borderRadius': '5px'})
        ]),
        html.Div([html.Label("Train-Test Percentage:"), dcc.Slider(id='training-percent', min=0.1, max=1, step=0.1,
                   marks={0.1: '10%', 1: '100%'},
                   value=0.8
                   )], style={'display': 'None', 'padding': '20'})
        ], style={'display': 'inline-block', 'vertical-align': 'top', 'width': '45%', 'padding': '10'}),
        
        #--------
        #Stores uploded data converted to df-to-json 
        html.Div(id='upload-data-df', style={'display': 'none'}),

        #---------
        #Footnote
        html.Div([dcc.Markdown(footnote_markdown)], 
                  style={'borderTop': 'thin lightgrey solid', 'padding': '10', 'fontSize': 'small'}),
        ], className='container'
)

#========================================================
# #APP INTERACTIONS BELOW
#========================================================       

#--------------------------------------------------------
#Show SVR Inputs
@app.callback(
    dash.dependencies.Output('column-options', 'style'),
    [dash.dependencies.Input('upload-data-df', 'children')])

def update_colblock(value):
    if value is not None:
        return {'display': 'block'}
    else:
        return {'display': 'none'}
    
#--------------------------------------------------------
#Show SVR Inputs
@app.callback(
    dash.dependencies.Output('svr-inputs', 'style'),
    [dash.dependencies.Input('model-dropdown', 'value')])

def update_svrblock(value):
    if 2 in value:
        return {'display': 'block', 'padding': '20 0'}
    else:
        return {'display': 'none'}

#-------------------------------------------------------- 
#Update graph based on filters
@app.callback(
    dash.dependencies.Output('plot-data', 'figure'),
    [dash.dependencies.Input('kernel-selector', 'value'),
     dash.dependencies.Input('cost-function', 'value'),
     dash.dependencies.Input('epsilon', 'value'),
     dash.dependencies.Input('upload-data-df', 'children'),
     dash.dependencies.Input('X-dropdown', 'value'),
     dash.dependencies.Input('y-dropdown', 'value'),
     dash.dependencies.Input('model-dropdown', 'value'),
     dash.dependencies.Input('hide-svr-tolerance', 'values'),
     dash.dependencies.Input('select-dataset', 'value'),
     dash.dependencies.Input('training-percent', 'value')])

def regression_model(kernel_selected, cost_function, epsilon, json_intermediate_data, x_column, y_column, model, svr_tolerance, default_dataset, training_percent):
    '''
    '''
    if default_dataset is not None and default_dataset != '':
        Regressor, Datasets = sample_data()
        df = Datasets[default_dataset]
        x_column = 'X'
        y_column = 'y'
        #X = df['X'].as_matrix().reshape(-1, 1)
        #y = df['y'].as_matrix()
    elif json_intermediate_data is not None:
        df = pd.read_json(json_intermediate_data, orient='split')[[x_column, y_column]]
        df = df[:100].sort_values([x_column])
    
    
    #df_train = df.sample(frac=training_percent).sort_values([x_column])
    #if training_percent < 1:
    #    df_test = df.drop(df_train.index).sort_values([x_column])
    #elif training_percent == 1:
    #    df_test = df_train
    
    X = df.sort_values(x_column)[x_column].as_matrix().reshape(-1, 1)
    y = df.sort_values(x_column)[y_column].as_matrix()
    
    #Test:Train Split functionality not available yet
    #---
    #X = df_train[x_column].as_matrix().reshape(-1, 1)
    #y = df_train[y_column].as_matrix()
    
    #X_test = df_test[x_column].as_matrix().reshape(-1, 1)
    #y_test= df_test[y_column].as_matrix()
    
    
    traces=[]
    #>>>Simple Linear Regression<<<
    if 1 in model:
        lr = linear_model.LinearRegression()
        lr.fit(X, y)
        y_predicted = lr.predict(X)
        name = "OLS Reg., R-sq={:.2%}".format(lr.score(X,y)) 
        
        traces.append(go.Scatter(
            x=X.ravel(),
            y=lr.predict(X),
            mode='lines', name=name))
    
    #>>>SVR<<<
    if 2 in model: 
        for kernels in kernel_selected:
            svr= SVR(kernel=kernels, C=float(cost_function), epsilon=float(epsilon))
            y_predicted = svr.fit(X, y).predict(X)
            name = "Support-Vector Reg. " + kernels.upper() + ", R-sq: {:.2%}".format(svr.score(X,y))
            
            if 1 in svr_tolerance:
                traces.append(go.Scatter(
                    x=X.ravel().tolist() + X.ravel().tolist()[::-1],
                    y=[y+float(epsilon) for y in y_predicted.tolist()] + [y-float(epsilon) for y in y_predicted.tolist()][::-1],
                    fill='tozeroy', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), name=kernels.upper() + ' tolerance'
                    ))
            
            traces.append(go.Scatter(
                x=X.ravel(),
                y=y_predicted,
                mode='lines', name=name))
    
    #>>>Robust Linear<<<
    if 3 in model:
        ransac = linear_model.RANSACRegressor()
        ransac.fit(X, y)
        inlier_mask_ransac = ransac.inlier_mask_
        outlier_mask_ransac = np.logical_not(inlier_mask_ransac)
        y_predicted = ransac.predict(X)
        name = "RANSAC Reg., R-sq={:.2%}".format(ransac.score(X[inlier_mask_ransac],y[inlier_mask_ransac]))
        
        traces.append(go.Scatter(
            x=X.ravel(),
            y=y_predicted,
            mode='lines', name=name))
    
    #>>>TheilSen Regression<<<
    if 4 in model:
        theilsen = linear_model.TheilSenRegressor()
        theilsen.fit(X, y)
        y_predicted = theilsen.predict(X)
        name = "Theilsen Reg., R-sq={:.2%}".format(theilsen.score(X,y))
        
        traces.append(go.Scatter(
            x=X.ravel(),
            y=y_predicted,
            mode='lines', name=name))
    
    #Add actual data points
    traces.append(go.Scatter(
                x=X.ravel(),
                y=y,
                mode='markers', name='Training Data'))
    
    #traces.append(go.Scatter(
    #            x=X_test.ravel(),
    #            y=y_test,
    #            mode='markers', name='Testing Data'))
        
    return {
        'data': traces,
        'layout': go.Layout(
            title="Regression Modeling on Single Variable",
            font=dict(family='sans-serif', size=12, color='#7f7f7f'),
            xaxis={'title': x_column},
            yaxis={'title': y_column},
            hovermode='closest',
            legend=dict(x=0, y=-0.5, traceorder='normal',
                            font=dict(family='sans-serif', size=12, color='#000'),
                            bgcolor='#E2E2E2', bordercolor='#FFFFFF', borderwidth=2
                            )
        )
    }

#-------------------------------------------------------- 
#Update X-Dropdown Filter
@app.callback(
    dash.dependencies.Output('X-dropdown', 'options'),
    [dash.dependencies.Input('upload-data-df', 'children')])
     
def update_dropdown_options_x(json_intermediate_data):    
    '''
    '''
    df = pd.read_json(json_intermediate_data, orient='split')
    column_options = []
    for column_name in df.columns:
        column_options.append({'label': column_name, 'value': column_name}) 
    return column_options

#-------------------------------------------------------- 
#Update X-Dropdown Filter
@app.callback(
    dash.dependencies.Output('y-dropdown', 'options'),
    [dash.dependencies.Input('upload-data-df', 'children')])
     
def update_dropdown_options_y(json_intermediate_data):    
    '''
    '''
    df = pd.read_json(json_intermediate_data, orient='split')
    column_options = []
    for column_name in df.columns:
        column_options.append({'label': column_name, 'value': column_name}) 
    return column_options

#-------------------------------------------------------- 
#Store the uploaded DF
@app.callback(dash.dependencies.Output('upload-data-df', 'children'),
              [dash.dependencies.Input('upload-data', 'contents'),
               dash.dependencies.Input('upload-data', 'filename'),
               dash.dependencies.Input('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
         return parse_contents(list_of_contents, list_of_names, list_of_dates)
     
#--------------------------------------------------------
#Append cool CSS style sheet
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})


if __name__ == '__main__':
    app.run_server(debug=True)