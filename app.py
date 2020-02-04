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
import textwrap as  tw
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.model_selection import train_test_split

#Dash and Plotly modules
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.exceptions import PreventUpdate
import dash_daq as daq
import plotly.graph_objs as go

#Custom modules
from parse_contents import parse_contents
from sample_datasets import sample_data

external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css",
                      "https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css",
                      "https://unpkg.com/tailwindcss@^1.0/dist/tailwind.min.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Regression Simulator'

server = app.server

subheading_markdown = '''
__Description__:
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
Created by: Aveedibya Dey | [Source Code](https://github.com/aveedibya/regressionSimulator) | [Contact Me/Provide Feedback](https://aveedibyadey.typeform.com/to/VKEs3v)
    '''

app.layout = html.Div(html.Div(children=[

        #---------
        #Heading
        dcc.Markdown(children='''#### Regression Model Simulator ''', className='h1 text-center'),

        #---------
        #Sub-heading
        html.Div([dcc.Markdown(subheading_markdown, className='h4 p-2 text-center')],
                  # style={'borderBottom': 'thin lightgrey solid', 'padding': '2'}
                  ),

        #---------
        #Chart
        html.Div([
                    html.Div(id='plot-train-data-div', children=dcc.Graph(id='plot-data'),
                             style={'display': ''}, className='my-2 mx-1 p-0 col border border-light shadow-sm'),
                    html.Div(id='plot-test-data-div', children=dcc.Graph(id='plot-test-data'),
                             style={'display': 'none'}, className='my-2 p-1 col-6 border border-light shadow-sm')
                 ], className='row'),

        #---------
        #Controls
        html.Div([
            html.Div([html.Div([html.Label('Select Regression Model:', className='h5'),
                               dcc.Dropdown(id='model-dropdown',
                                   options=[
                                {'label': 'Simple Linear Regression', 'value': 1},
                                {'label': 'Support Vector Regression', 'value': 2},
                                {'label': 'RANSAC Linear Regression', 'value': 3},
                                {'label': 'Theilsen Linear Regression', 'value': 4},
                                {'label': 'Huber Regression', 'value': 5}],
                                placeholder="Select Regression Model",
                                multi=True,
                                className='form-group')]),
                      #------------
                      #SVR Options
                      html.Div(id='svr-inputs', children=[
                              html.Div([
                                      html.Label('Select SVR Kernel Function:'),
                                      dcc.Dropdown(id='kernel-selector',
                                            options=[
                                                {'label': 'Radial Basis Function', 'value': 'rbf'},
                                                {'label': 'Linear', 'value': 'linear'},
                                                {'label': 'Polynomial', 'value': 'poly'},
                                                {'label': 'Sigmoid', 'value': 'sigmoid'}
                                                ], placeholder="Select SVR Kernel", multi=True)], className='col-12'),
                             html.Div([
                                        html.Div([
                                                 html.Label('Penalty Ratio, C:', className='col-6'),
                                                 dcc.Input(id='cost-function', type='number', step=0.01, placeholder='C=1.0', className='col-6')
                                                 ], className='col-12 p-0 mt-1'),
                                        html.Div([
                                                 html.Label('Epsilon for Support-Vector:', className='col-6'),
                                                 dcc.Input(id='epsilon', type='number', step=0.01, placeholder='epsilon=0.5', className='col-6')
                                                 ], className='col-12 p-0 mt-1'),
                                        dcc.Checklist(id='hide-svr-tolerance', options=[{'label': ' Show SVR Tolerance', 'value':1}], value=[1], className='col-12 mt-1')
                                     ], className='col-12 mt-2'),
                            ],
                    style={'background-color': '', 'padding': '20', 'display': 'none'},
                    className='alert alert-light rounded-lg shadow-sm')
            ], style={'display': 'inline-block', 'vertical-align': 'top', 'padding': '10', 'width': '45%'}, className='col-md'),
            #--------
            #Block Below Graph
            html.Div([
                html.Label('Select Data and Regression Columns:', className='h5 col-12'),

                #--------
                #Column Selection
                html.Div(id='column-options', children=[html.Label('Independent Variable, X:'),
                          dcc.Dropdown(id='X-dropdown'),
                          html.Label('Dependent Variable, y:'),
                          dcc.Dropdown(id='y-dropdown')
                          ], className='col-12 alert alert-info rounded-lg shadow-sm'),
                #--------
                #Data Selection Elements
                html.Div([html.Div(dcc.Upload(id='upload-data',
                                              children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                                              className='p-4 text-center h6'),
                                    className='col-12 alert alert-dark rounded-lg shadow-sm'),

                    html.Div([
                              html.Div([
                                    html.Label(id='row-trimmer-label', children='Trim Training Data:', className='col-12'),
                                    dcc.Slider(id='row-trimmer', min=0.01, max=1, value=1, step=0.01, marks={}, updatemode='mouseup', className='col-12 p-0')
                                    ], className='col-7'),
                              html.Div([
                                    html.Label('Split Train/Test:', className='col-6 text-center p-0 m-0'),
                                    daq.BooleanSwitch(id='train-test-split', on=False, className='col-6 p-0 m-0'),
                                    ], className='col-5')
                            ], className='row mb-2'),

                    html.Div(id='output-data-upload', className='col-12 m-0 p-0'),

                    html.Div([html.Label(dcc.Markdown('*__OR__ Select Built-in Sample Data:*')),
                              html.Div(dcc.Dropdown(id='select-dataset',
                                   options=[
                                       {'label': 'No Outliers', 'value': 'df_normal'},
                                       {'label': 'Small Outliers in x', 'value': 'df_x_errors'},
                                       {'label': 'Small Outliers in y', 'value': 'df_y_errors'},
                                       {'label': 'Large Outliers in x', 'value': 'df_x_large_errors'},
                                       {'label': 'Large Outliers in y', 'value': 'df_y_large_errors'},
                                       {'label': 'Sine Function with Noises & Outliers', 'value': 'df_rbf_eps'},
                                       {'label': 'Sine Function', 'value': 'df_rbf_c'}
                                   ],
                                   placeholder='Built-in Dataset Not Selected',
                                   value=''
                                   ), style={})
                            ],
                            # style={'backgroundColor': '#ebebe0', 'margin-top': '5px', 'padding': '10', 'borderRadius': '5px'},
                            className='col-12 alert alert-light rounded-lg shadow-sm')
                ], className='container')
            ], style={'display': 'inline-block', 'vertical-align': 'top', 'width': '45%', 'padding': '10'}, className='col-md'),
        ], className='row my-4'),
        #--------
        #Stores UPLOADED data converted to df-to-json
        html.Div(id='upload-data-df', style={'display': 'none'}),

        #--------
        #Stores uploded SAMPLE data converted to df-to-json
        html.Div(id='sample-data-store', style={'display': 'none'}),

        #---------
        #Footnote
        html.Div(dcc.Markdown(footnote_markdown),
                  style={'borderTop': 'thin lightgrey solid'}, className='mt-4 pt-2 text-muted'),
        ], className='container my-2'
), className='bg-light py-4')

#========================================================
# #APP INTERACTIONS BELOW
#========================================================

#Show Test Train Split
@app.callback([dash.dependencies.Output('row-trimmer-label', 'children'),
               dash.dependencies.Output('plot-test-data-div', 'style'),
               dash.dependencies.Output('plot-train-data-div', 'className')],
               [dash.dependencies.Input('row-trimmer', 'value'),
                dash.dependencies.Input('train-test-split', 'on')])

def test_train_split(test_sample_size, test_train_split_on):
    if test_sample_size < 1:
        row_trimmer_label = 'Training On: {:.1%}'.format(test_sample_size)
    else:
        raise PreventUpdate

    if test_train_split_on==True:
        return row_trimmer_label, {'display': ''}, 'my-2 p-1 mx-0 col-6 border border-light shadow-sm'
    else:
        return row_trimmer_label, {'display': 'none'}, 'my-2 mx-0 p-1 col border border-light shadow-sm'


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
    if value is None:
         raise PreventUpdate
    else:
        if 2 in value:
            return {'display': 'block', 'padding': '20 0'}
        else:
            return {'display': 'none'}

#--------------------------------------------------------
#Update graph based on filters
@app.callback(
    [dash.dependencies.Output('plot-data', 'figure'),
     dash.dependencies.Output('plot-test-data', 'figure')],
    [dash.dependencies.Input('kernel-selector', 'value'),
     dash.dependencies.Input('cost-function', 'value'),
     dash.dependencies.Input('epsilon', 'value'),
     dash.dependencies.Input('upload-data-df', 'children'),
     dash.dependencies.Input('X-dropdown', 'value'),
     dash.dependencies.Input('y-dropdown', 'value'),
     dash.dependencies.Input('model-dropdown', 'value'),
     dash.dependencies.Input('hide-svr-tolerance', 'value'),
     dash.dependencies.Input('select-dataset', 'value'),
     dash.dependencies.Input('row-trimmer', 'value')
     ])

def regression_model(kernel_selected, cost_function, epsilon, json_intermediate_data, x_column, y_column, model, svr_tolerance, default_dataset, rows_selected):
    '''
    '''
    if x_column is None and y_column is None:
         raise PreventUpdate
    else:
        if model is None:
            model = []

        if default_dataset is not None and default_dataset != '':
            #Regressors, Datasets = sample_data()
            #df = Datasets[default_dataset]
            df = pd.read_json(json_intermediate_data, orient='split')[[x_column, y_column]]
            # df = df[:int(rows_selected*df.shape[0])].sort_values([x_column])
            # print(df)
            x_column = 'X'
            y_column = 'y'
        elif json_intermediate_data is not None:
            df = pd.read_json(json_intermediate_data, orient='split')[[x_column, y_column]]
            # df = df[:int(rows_selected*df.shape[0])].sort_values([x_column])

        train_df, test_df = train_test_split(df, test_size=1-rows_selected)

        #Test:Train Split functionality
        #---
        X = train_df.sort_values(x_column)[x_column].as_matrix().reshape(-1, 1)
        y_df = train_df.sort_values(x_column)[y_column]
        y=y_df.as_matrix()

        # Prepare Test data ---
        X_test = test_df.sort_values(x_column)[x_column].as_matrix().reshape(-1, 1)
        y_df_test = test_df.sort_values(x_column)[y_column]
        y_test=y_df_test.as_matrix()

        traces=[]
        traces_test=[]

        #>>>Simple Linear Regression<<<
        if 1 in model:
            lr = linear_model.LinearRegression()
            lr.fit(X, y)
            y_predicted = lr.predict(X)
            print(lr.get_params())
            name = "OLS Reg., R-sq={:.2%}".format(lr.score(X,y))

            traces.append(go.Scatter(
                x=X.ravel(),
                y=lr.predict(X),
                mode='lines', name=name))

            if rows_selected < 1:
                name_test = "OLS Reg., R-sq={:.2%}".format(lr.score(X_test,y_test))
                traces_test.append(go.Scatter(
                    x=X_test.ravel(),
                    y=lr.predict(X_test),
                    mode='lines', name=name_test))

        #>>>SVR<<<
        if 2 in model:
            for kernels in kernel_selected:
                svr= SVR(kernel=kernels, C=float(cost_function), epsilon=float(epsilon))
                svr_model = svr.fit(X, y)
                y_predicted = svr_model.predict(X)
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

                if rows_selected < 1:
                    name_test = "Support-Vector Reg. " + kernels.upper() + ", R-sq: {:.2%}".format(svr.score(X_test,y_test))
                    traces_test.append(go.Scatter(
                        x=X_test.ravel(),
                        y=svr_model.predict(X_test),
                        mode='lines', name=name_test))

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

            if rows_selected < 1:
                name_test = "RANSAC Reg., R-sq={:.2%}".format(ransac.score(X_test,y_test))
                traces_test.append(go.Scatter(
                    x=X_test.ravel(),
                    y=ransac.predict(X_test),
                    mode='lines', name=name_test))


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

            if rows_selected < 1:
                name_test = "Theilsen Reg., R-sq={:.2%}".format(theilsen.score(X_test,y_test))
                traces_test.append(go.Scatter(
                    x=X_test.ravel(),
                    y=theilsen.predict(X_test),
                    mode='lines', name=name_test))

        #>>>Huber Regression<<<
        if 5 in model:
            huber = linear_model.HuberRegressor()
            huber.fit(X, y)
            y_predicted = huber.predict(X)
            name = "Huber Reg., R-sq={:.2%}".format(huber.score(X,y))

            traces.append(go.Scatter(
                x=X.ravel(),
                y=y_predicted,
                mode='lines', name=name))

            if rows_selected < 1:
                name_test = "Huber Reg., R-sq={:.2%}".format(huber.score(X_test,y_test))
                traces_test.append(go.Scatter(
                    x=X_test.ravel(),
                    y=huber.predict(X_test),
                    mode='lines', name=name_test))



        #Add actual data points
        traces.append(go.Scatter(
                    x=X.ravel(),
                    y=y_df,
                    mode='markers', name='Training Data'))

        if rows_selected < 1:
            traces_test.append(go.Scatter(
                        x=X_test.ravel(),
                        y=y_df_test,
                        mode='markers', name='Test Data'))

        return {
            'data': traces,
            'layout': go.Layout(
                title="Regression Modeling on Single Variable (Train: {:.1%})".format(rows_selected),
                font=dict(family='sans-serif', size=10, color='#7f7f7f'),
                xaxis={'title': x_column},
                yaxis={'title': y_column},
                hovermode='closest',
                legend=dict(x=0, y=-0.5, traceorder='normal',
                                font=dict(family='sans-serif', size=12, color='#000'),
                                bgcolor='#E2E2E2', bordercolor='#FFFFFF', borderwidth=2
                                )
            )
        }, {
        'data': traces_test,
        'layout': go.Layout(
            title="Regression Modeling on Single Variable (Test: {:.1%})".format(1-rows_selected),
            font=dict(family='sans-serif', size=10, color='#7f7f7f'),
            xaxis={'title': x_column},
            yaxis={'title': y_column},
            hovermode='closest',
            height=450,
            # width=500,
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
    if json_intermediate_data is None:
         raise PreventUpdate
    else:
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
    if json_intermediate_data is None:
         raise PreventUpdate
    else:
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
               dash.dependencies.Input('upload-data', 'last_modified'),
               dash.dependencies.Input('select-dataset', 'value')])
def update_output(list_of_contents, list_of_names, list_of_dates, sample_data_selected):
    if list_of_contents is not None:
         return parse_contents(list_of_contents, list_of_names, list_of_dates)
    elif sample_data_selected is not None and sample_data_selected != '':
         return sample_data()[1][sample_data_selected].to_json(date_format='iso', orient='split')

#--------------------------------------------------------
#Append cool CSS style sheet
#app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
#app.scripts.config.serve_locally = False


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=True)
