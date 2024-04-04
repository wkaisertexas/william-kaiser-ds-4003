#!/usr/bin/env python
# coding: utf-8

# # Breaking Down UVA Course Reviews
# 
# By: William Kaiser
# 
# ## Layout
# 
# The layout for this site was designed in [Figma](https://www.figma.com/file/PCZFfmRXn0e6720BjxNAtg/Misc-Images?type=design&node-id=539-5&mode=design&t=4der6LRc6aprI2Sf-0) and designed to be responsive and intent-based.
# 
# ![Layout](./imgs/layout.png)
# 
# However, after discussion with fellow class members and looking at [Python Graph Gallery](https://python-graph-gallery.com/) I decided to use a [Spider / Radar](https://python-graph-gallery.com/radar-chart/) plot as well as the more traditional bar charts.

# # Data Loading
# 
# The data for this analysis comes from [theCourseForum](https://thecourseforum.com/) and a direct database connection was used. 
# 
# For more information about data provenance, collection, formatting, and cleaning, please see [Sprint 2: Data](./sprints/sprint2-data.ipynb).

# In[167]:


# Generic imports
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import dash_table, dcc, Dash, html, callback, Input, Output, State
from typing import List, Dict, TypedDict
import os


# In[168]:


# Loading the data
df = pd.read_csv("./data.csv")
df['course_level'] = pd.to_numeric(df['course_number']).apply(lambda x: x // 1000)
df


# In[169]:


# Printing the columns
print("\n".join(df.columns.to_list()))


# ## Making the Spider Plot
# 
# The goal of this section is for a specified course to be selected and then a radar plot to be generated based on the reviews for that course.

# In[170]:


REVIEW_COMPONENTS = {
    "instructor_rating": "Instructor Rating",
    "difficulty": "Difficulty",
    "recommendability": "Recommendability",
    "enjoyability": "Enjoyability",
    # "hours_per_week": "Hours Per Week",
    "amount_reading": "Amount Reading",
    "amount_writing": "Amount Writing",
    "amount_group": "Amount Groupwork",
    "amount_homework": "Amount Homework",
}

# making each column into a numeric column
for column in REVIEW_COMPONENTS.keys():
    df[column] = pd.to_numeric(df[column], errors="coerce")

COURSE = "CS 3100"

def course_components(course_pneumonic_and_number: str) -> (str, int):
    """
    Turns a human readable course pneumonic and number into a database-searchable tuple
    """
    course_pneumonic_and_number = (
        course_pneumonic_and_number.strip().upper().replace("  ", " ").split(" ")
    )
    pneumoic, number = course_pneumonic_and_number
    print(pneumoic, number)
    return pneumoic, int(number)


def get_course_summary_ratings(pneumonic: str, number: int) -> pd.DataFrame:
    """
    Gets the ratings for the course
    """
    pneumonic_mask = df["mnemonic"] == pneumonic
    number_mask = df["course_number"] == number
    relevant = df[pneumonic_mask & number_mask]

    mean = relevant[REVIEW_COMPONENTS.keys()].mean()
    std = relevant[REVIEW_COMPONENTS.keys()].std()

    frame = pd.DataFrame(
        {"mean": mean.to_list(), "std": std.to_list(), "category": mean.index.to_list()}
    )
    frame['course'] = frame['mean'].apply(lambda x: f"{pneumonic} {number}")
    return frame


get_course_summary_ratings(*course_components(COURSE))


# In[171]:


# Creating a spider plot for the course
# source: https://python-graph-gallery.com/571-radar-chart-with-plotly/
# Getting many plot with px
# source: https://stackoverflow.com/questions/56727843/how-can-i-create-subplots-with-plotly-express
# note: this sucks


def course_axis(
    frame: pd.DataFrame, course: str = None, include: str = "ratings"
) -> px.line_polar:
    """
    Makes a polar axis to compare courses
    """
    if "duration" in include:
        frame = frame[frame["category"].apply(lambda name: name.find("amount") != -1)]
    if "rating" in include:
        frame = frame[frame["category"].apply(lambda name: name.find("amount") == -1)]

    fig = go.Figure()
    frame['category'] = frame['category'].replace(REVIEW_COMPONENTS)
    for course in frame["course"].unique():
        rel_frame = frame[frame["course"] == course]
        graph_object = go.Scatterpolar(
            r=rel_frame["mean"], theta=rel_frame["category"], fill="toself",
             name=course.upper()
        )

        fig.add_trace(graph_object)

    return fig


cs_3100 = get_course_summary_ratings(*course_components(COURSE))
cs_2130 = get_course_summary_ratings(*course_components("CS 2130"))
cs_3130 = get_course_summary_ratings(*course_components("CS 3130"))
ds_4003 = get_course_summary_ratings(*course_components("DS 4003"))


rows = pd.concat([cs_3100, cs_2130, cs_3130, ds_4003])

course_axis(rows, course="CS 3100")


# ## Excel-Style Filterable Table
# 
# The goal of this section is to allow user filtering of tables in an excel like manner. 
# 
# The data in the section will be similar to [the spider plot](#making-the-spider-plot), however the format will be a bit more formal.
# 
# ![Excel-style Table](./imgs/excel-table.png)
# 
# **Guide:** https://dash.plotly.com/datatable

# In[172]:


# Defining columns of interest
SAMPLE_COURSES = ["CS 3100", "CS 2130", "CS 3130", "DS 4003"]

# additional table components
TABLE_COMPONENTS = {
    'average': 'Average',
}

course_components_to_agg = {**TABLE_COMPONENTS, **REVIEW_COMPONENTS}
course_components_to_agg


# In[173]:


def get_data_for_course_comparison_table(courses: List) -> pd.DataFrame:
    """
    Gets relevant course_components to aggregate for a list of courses
    """
    print(courses)

    # filtering the data down to the relevant frame
    mask = df["mnemonic"] == -1
    for course in courses:
        pneumonic, number = course
        mask = mask | ((df["mnemonic"] == pneumonic) & (df["course_number"] == number))
    
    relevant = df[mask]

    grouped = relevant.groupby(["mnemonic", "course_number"]).agg({ key: 'mean' for key in course_components_to_agg.keys() }).reset_index()

    return grouped

course_tuples = [course_components(course) for course in SAMPLE_COURSES]
course_data = get_data_for_course_comparison_table(course_tuples)
course_data


# In[174]:


# making the table in plotly
# source: https://plotly.com/python/table/

course_components_to_agg = {
    **course_components_to_agg,
    "course_number": "Course #",
    "mnemonic": "Mnemonic",
}


def get_cells(course_data: pd.DataFrame) -> List[List[any]]:
    """
    Gets the cells in the data table
    """
    return course_data.round(2).to_numpy().transpose().tolist()


pretty_column_names = list(
    map(
        lambda col: f"<b>{course_components_to_agg.get(col, col).replace(' ', '<br>')}</b><br>",
        course_data.columns.to_list(),
    )
)


table = go.Figure(
    data=[
        go.Table(
            header=dict(
                values=pretty_column_names,
                line_color="darkslategray",
                fill_color="royalblue",
                align=["left", "center"],
                font=dict(color="white", size=12),
                height=40,
            ),
            cells=dict(
                values=get_cells(course_data),
                line_color="darkslategray",
                fill=dict(color=["paleturquoise", "white"]),
                align=["left", "center"],
                font_size=12,
                height=30,
            ),
        )
    ]
)

table  # note: I am pretty happy with this. I would like to add more interactivity here.


# In[175]:


# making a source dropdown
# source: https://dash.plotly.com/dash-core-components/dropdown
# df['course_number'] = df['course_number'].apply(int)
df['course_number'].fillna(0, inplace=True)
df['name'] = df['mnemonic'] + " " + df['course_number'].astype(str)
df['name']


# In[176]:


# adding source dropdowns here
course_mneumoic_dropdown = dcc.Dropdown(
    id='course_dropdown',
    options=[{'label': str(course).replace(".0", ""), 'value': str(course).replace(".0", "")} for course in df['name'].unique()],
    value='CS 3100',
    multi=True
)


# # Breaking Down Instructor Rating
# 
# ![Correlating Instructor Reviews](./imgs/correlating_reviews.png)

# In[177]:


# the goal for this is to create the functions which can control for a  particular factor
CHECKBOX_SELECTORS = {
    'average': 'GPA',
    # 'course_level': 'Course Level (1k, 2k, ...)',
    'difficulty' : 'Difficulty',
    'hours_per_week': 'Hours Worked Per Week',
    'amount_group': 'Group Work Per Week'
}


# In[178]:


# making a bunch of checkboxes in plotly
checklist = dcc.Checklist(
    options=[
        {'label': v, 'value': k} for k, v in CHECKBOX_SELECTORS.items()
    ],
    value=[],
    id='features_to_plot'
)


# In[179]:


# creating a histogram of course reviews
course_reviews = px.histogram(
    df.groupby(["mnemonic", "course_number"]).agg({"instructor_rating": "mean"}),
    x="instructor_rating",
)
course_reviews


# ### Making a regression

# In[180]:


import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# making a linear regression model
model = LinearRegression()

columns = ['difficulty', 'instructor_rating', 'hours_per_week']
# getting the data
new_columns = ['average', 'difficulty', 'hours_per_week', 'amount_group']
df.dropna(subset=new_columns, inplace=True)
X = df[new_columns].values.reshape(-1, len(new_columns))
Y = df['instructor_rating'].values.reshape(-1, 1)

# fitting the model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model.fit(X_train, Y_train)

print(model.coef_, model.intercept_)


# In[181]:


@callback(
    Output('correlation_plot', 'figure'),
    Input('features_to_plot', 'value')
)
def make_correlation_plot(columns): 
    """
    Makes a bar chart of the correlation between columns
    """
    # uses the data from the model
    data = zip(new_columns, model.coef_[0])
    print(data)
    # filter the data by which the column is in the columns
    column_renamer = {
        'average': 'GPA',
        'difficulty': 'Difficulty',
        'hours_per_week': 'Hours Worked Per Week',
        'amount_group': 'Group Work Per Week'
    }

    x_labels = []
    y_values = []
    for col, r in data:
        if col in columns:
            x_labels.append(column_renamer.get(col, col))
            y_values.append(r)
    
    print(x_labels, y_values)
    
    fig = px.bar(x=x_labels, y=y_values, labels= {'x': 'Feature', 'y': 'Correlation Coefficient', **column_renamer})
    return fig

make_correlation_plot(new_columns)


# # Final Dashboard Created

# In[182]:


# defining callbacks
@callback(
    Output(component_id="course_axis", component_property="figure"),
    (Input(component_id="course_dropdown", component_property="value"))
)
def update_course_axis(course: str):
    """
    Updates the course axis
    """
    if course is str: # if the course is a string
        course = [course]
    
    summary = pd.concat([get_course_summary_ratings(*course_components(course)) for course in course])
    
    # rows = pd.concat([cs_3100, cs_2130, cs_3130, ds_4003])

    # course_data = get_course_summary_ratings(*course_components(course))
    return course_axis(summary, course=course)

@callback(
    Output(component_id="course_hours", component_property="figure"),
    (Input(component_id="course_dropdown", component_property="value"))
)
def update_course_hours(course: str):
    """
    Updates the course hours
    """
    if course is str:
        course = [course]
    
    summary = pd.concat([get_course_summary_ratings(*course_components(course)) for course in course])
    return course_axis(summary, course=course, include="duration")


### Training a linear model to predict the instructor rating based on (GPA, Course Level, Difficulty, Hours Worked Per Week, Group Work Per Week)


# In[ ]:


# Making an app to display everything
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

### HEADER ###
header = html.Div([
    html.H1("Course Review Explorer"),
])

### BREAKING DOWN RATINGS ###
course_filter_header = html.Div([
    html.H1("Breaking down ratings"),
    html.P("Examine course reviews for a particular class"),
])

course_filter_row = html.Div([
     dcc.Graph(figure=course_axis(rows, course="CS 3100"), id="course_axis", className="one-half column"),
     dcc.Graph(figure=course_axis(rows, course="CS 2130"), id="course_hours", className="one-half column"),
     dcc.Graph(figure=table, id="course_table", className="one-half column"),
], className="row")

course_filter_section = html.Div([
    course_filter_header,
    course_mneumoic_dropdown,
    course_filter_row,
])

### CORRELATING INSTRUCTOR REVIEWS ###
controlling_factors = html.Div([
    html.H1("Correlating Instructor Reviews"),
    html.Div("Control for the following factors"),
    checklist,
], className="one-third column")

distribution_of_reviews = html.Div([
    html.H1("Distribution of Residuals"),
    html.P("What's left over?"),
    # dcc.Graph(figure=course_reviews, id="course_reviews"),
], className="one-third column")

predictive_power_of_features = html.Div([
    html.H1("Predictive Power of Features"),
    html.P("How well do these features predict the instructor rating?"),
    dcc.Graph(id="correlation_plot"),
], className="one-third column")

correlating_instructor_reviews = html.Div([
    controlling_factors,
    distribution_of_reviews,
    predictive_power_of_features
], className="row")

### FINAL LAYOUT ###
app.layout = html.Div([
   header,
   course_filter_section,
   correlating_instructor_reviews,
], className="container")

server = app.server

if __name__ == '__main__':
    app.run_server(jupyter_mode='tab')

