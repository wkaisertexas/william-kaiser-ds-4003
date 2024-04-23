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

# In[565]:


# imports
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from openai import OpenAI
import plotly.graph_objects as go
import plotly.express as px
from dash import dash_table, dcc, Dash, html, callback, Input, Output
from typing import List
import os

tqdm.pandas()


# In[566]:


# Loading the data
df = pd.read_csv("./data.csv")
df['course_level'] = pd.to_numeric(df['course_number']).apply(lambda x: x // 1000)
df


# In[567]:


# Printing the columns
print("\n".join(df.columns.to_list()))


# ## Making the Spider Plot
# 
# The goal of this section is for a specified course to be selected and then a radar plot to be generated based on the reviews for that course.

# In[568]:


REVIEW_COMPONENTS = {
    "instructor_rating": "Instructor\nRating",
    "difficulty": "Difficulty",
    "recommendability": "Recommendability",
    "enjoyability": "Enjoyability",
    # "hours_per_week": "Hours Per Week",
    "amount_reading": "Amount\nReading",
    "amount_writing": "Amount\nWriting",
    "amount_group": "Amount\nGroupwork",
    "amount_homework": "Amount\nHomework",
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


# In[569]:


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


rows = pd.concat([cs_3100, cs_2130, cs_3130])

# course_axis(rows, course="CS 3100")


# ## Excel-Style Filterable Table
# 
# The goal of this section is to allow user filtering of tables in an excel like manner. 
# 
# The data in the section will be similar to [the spider plot](#making-the-spider-plot), however the format will be a bit more formal.
# 
# ![Excel-style Table](./imgs/excel-table.png)
# 
# **Guide:** https://dash.plotly.com/datatable

# In[570]:


# Defining columns of interest
SAMPLE_COURSES = ["CS 3100", "CS 2130", "CS 3130", "DS 4003"]

# additional table components
TABLE_COMPONENTS = {
    'average': 'Average',
}

course_components_to_agg = {**TABLE_COMPONENTS, **REVIEW_COMPONENTS}
course_components_to_agg


# In[571]:


def get_data_for_course_comparison_table(courses: List) -> pd.DataFrame:
    """
    Gets relevant course_components to aggregate for a list of courses
    """
    # filtering the data down to the relevant frame
    mask = df["mnemonic"] == -1
    for course in courses:
        pneumonic, number = course
        mask = mask | ((df["mnemonic"] == pneumonic) & (df["course_number"] == number))

    grouped = df[mask].groupby(["mnemonic", "course_number"]).agg({ key: 'mean' for key in course_components_to_agg.keys() }).reset_index()

    return grouped

course_tuples = [course_components(course) for course in SAMPLE_COURSES]
specific_data = get_data_for_course_comparison_table(course_tuples)
specific_data


# In[572]:


# making the table in plotly
# source: https://plotly.com/python/table/

new_course_components_to_agg = {
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
        lambda col: f"<b>{new_course_components_to_agg.get(col, col).replace(' ', '<br>')}</b><br>",
        specific_data.columns.to_list(),
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
                values=get_cells(specific_data),
                line_color="darkslategray",
                fill=dict(color=["paleturquoise", "white"]),
                align=["left", "center"],
                font_size=12,
                height=30,
            ),
        )
    ]
)

# table  # note: I am pretty happy with this. I would like to add more interactivity here.


# In[573]:


# making a source dropdown
# source: https://dash.plotly.com/dash-core-components/dropdown
# df['course_number'] = df['course_number'].apply(int)
df['course_number'].fillna(0, inplace=True)
df['name'] = df['mnemonic'] + " " + df['course_number'].astype(str)
df['name']


# In[574]:


# adding source dropdowns here
course_mneumoic_dropdown = dcc.Dropdown(
    id='course_dropdown',
    options=[{'label': str(course).replace(".0", ""), 'value': str(course).replace(".0", "")} for course in df['name'].unique()],
    value=['CS 3100', 'CS 2130', 'CS 3130', 'DS 4003'],
    multi=True
)


# # Breaking Down Instructor Rating
# 
# ![Correlating Instructor Reviews](./imgs/correlating_reviews.png)

# In[575]:


# the goal for this is to create the functions which can control for a  particular factor
CHECKBOX_SELECTORS = {
    'average': 'GPA',
    # 'course_level': 'Course Level (1k, 2k, ...)',
    'difficulty' : 'Difficulty',
    'hours_per_week': 'Hours Worked Per Week',
    'amount_group': 'Group Work Per Week'
}


# In[576]:


# making a bunch of checkboxes in plotly
checklist = dcc.Checklist(
    options=[
        {'label': v, 'value': k} for k, v in CHECKBOX_SELECTORS.items()
    ],
    value=[],
    id='features_to_plot'
)


# In[577]:


# creating a histogram of course reviews
course_reviews = px.histogram(
    df.groupby(["mnemonic", "course_number"]).agg({"instructor_rating": "mean"}),
    x="instructor_rating",
)

# course_reviews


# ### Making a regression

# In[578]:


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


# In[579]:


@callback(
    Output("correlation_plot", "figure"),
    Input("features_to_plot", "value"),
    prevent_initial_call=True,  # FOR SOME REASON, this being called initially bricks the entire application
)
def make_correlation_plot(columns):
    """
    Makes a bar chart of the correlation between columns
    """
    # uses the data from the model
    data = zip(new_columns, model.coef_[0])

    # filter the data by which the column is in the columns
    column_renamer = {
        "average": "GPA",
        "difficulty": "Difficulty",
        "hours_per_week": "Hours Worked Per Week",
        "amount_group": "Group Work Per Week",
    }

    if isinstance(columns, str):
        columns = [columns]

    if len(columns) == 0:
        return

    if columns[0] == "":
        return  # we converted an empty string

    x_labels = []
    y_values = []
    for col, r in data:
        if col in columns:
            x_labels.append(column_renamer.get(col, col))
            y_values.append(r)

    fig = px.bar(
        x=x_labels,
        y=y_values,
        title="Predictive Power of Each Feature",
        labels={"x": "Feature", "y": "Linear Regression Strength", **column_renamer},
    )

    fig.update_layout(margin=dict(b=20, l=20, r=20, t=40))
    return fig


make_correlation_plot(new_columns)


# In[580]:


@callback(Output("review_residuals", "figure"), Input("features_to_plot", "value"))
def make_residual_plot(columns):
    """
    Makes a bar chart of the correlation between columns
    """
    data = zip(new_columns, model.coef_[0])

    # filter the data by which the column is in the columns
    column_renamer = {
        "average": "GPA",
        "difficulty": "Difficulty",
        "hours_per_week": "Hours Worked Per Week",
        "amount_group": "Group Work Per Week",
    }

    # Getting the relevant columns
    x_labels = []
    y_values = []
    for col, r in data:
        if col in columns:
            x_labels.append(column_renamer.get(col, col))
            y_values.append(r)

    # Creating a data frame copy
    avg_course_ratings = (
        df.groupby(["mnemonic", "course_number"])[new_columns + ["instructor_rating"]]
        .mean()
        .reset_index()
    )

    new_df = avg_course_ratings.copy(deep=True)
    for col in new_columns:
        if col not in columns:
            new_df[col] = new_df[col].apply(lambda x: 0)

    # Making the residuals
    residuals = (
        pd.Series(model.predict(new_df[new_columns]).flatten())
        - new_df["instructor_rating"]
    )

    residuals = residuals.dropna()

    # Making a detailed histogram of residuals
    fig = px.histogram(
        x=residuals,
        title="Student's Review Residuals",
        nbins=100,
        labels={
            "count": "# of Reviews",  # for some reason this does not work
            "x": "Residual Review\n(what's left over)",
        },
    )

    fig.update_layout(yaxis_title="# of Courses", margin=dict(b=20, l=20, r=20, t=40))

    # making a plot of reviews
    return fig


# make_residual_plot([])


# In[581]:


# getting measures of importance
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, Y_test, n_repeats=10, random_state=42)

X_train


# ## Building Semantic Search
# 
# Semantic search will take place with an OpenAI client and a embedding model. This will use a pinecone database for textual similarity.
# 
# ![Semantic Search](./imgs/semantic-search.png)

# In[582]:


# getting and saving course reviews
course_reviews = df.groupby(['title', 'description']).agg({'instructor_rating': 'mean'}).reset_index()

def make_prompt(row):
    """
    Makes a prompt for the user to ask the course
    """

    return f"""course name {row['title']} ({row['description']}) has an average rating of {row['instructor_rating']}"""

course_reviews['prompt'] = course_reviews.apply(make_prompt, axis=1)
course_reviews['prompt']


# In[583]:


# getting the embeddings from OpenAI
# from: https://platform.openai.com/docs/guides/embeddings/use-cases

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# course_reviews['ada_embedding'] = course_reviews['prompt'].progress_apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
# course_reviews.to_csv('data_1k_embeddings.csv', index=False)


# In[584]:


from pinecone import Pinecone, ServerlessSpec
# import os

# initialize connection to pinecone (get API key at app.pc.io)
api_key = os.environ.get('PINECONE_API_KEY') or '6b801a89-8fad-44bd-a8f5-1c2be8a9d208'

# configure client
pc = Pinecone(api_key=api_key)
spec = ServerlessSpec(cloud='aws', region='us-west-2')

# try:
#     pc.create_index(
#         name="course-reviews-1k", 
#         dimension=1536, 
#         metric="euclidean",
#         spec=spec
#     )
# except:
#     pass # index already created


# In[585]:


index = pc.Index("course-reviews-1k")

# def force_ascii(string: str) -> str:
#     """
#     Forces a string to be ascii
#     """
#     return string.encode('ascii', errors='ignore').decode('ascii')

# # making the things to upsert
# upsertion_reviews = [
#     {"id": force_ascii(item["title"]), "values": item["ada_embedding"]}
#     for item in course_reviews.to_dict(orient="records")
# ]

# # upload loop
# for i in range(0, len(upsertion_reviews), 100):
#     print(f"Uploading {i} to {i + 100}")
#     index.upsert(vectors=upsertion_reviews[i:i + 100])


# # Callbacks
# 
# Centrally locates all of the callback code for the Dash app.

# In[586]:


# making a dash table so that things work off the cuff
@callback(
    Output('course_table', 'data', allow_duplicate=True),
    Input('course_dropdown', 'value'),
    prevent_initial_call=True,
)
def update_table_data(data) -> pd.DataFrame:
    """
    Updates the data table to show the dropdown values
    """
    if isinstance(data, str):
        data = [data]
    
    course_tuples = [course_components(course) for course in data]

    course_data = get_data_for_course_comparison_table(course_tuples)

    course_data['course_number'] = course_data['course_number'].apply(int).apply(str)
    course_data['average'] = course_data['average'].apply(lambda x: f"{x:.2f}")
    course_data['amount_reading'] = course_data['amount_reading'].apply(lambda x: f"{x:.2f}")
    course_data['amount_writing'] = course_data['amount_writing'].apply(lambda x: f"{x:.2f}")
    course_data['amount_group'] = course_data['amount_group'].apply(lambda x: f"{x:.2f}")
    course_data['amount_homework'] = course_data['amount_homework'].apply(lambda x: f"{x:.2f}")
    course_data['instructor_rating'] = course_data['instructor_rating'].apply(lambda x: f"{x:.2f}")
    course_data['difficulty'] = course_data['difficulty'].apply(lambda x: f"{x:.2f}")
    course_data['recommendability'] = course_data['recommendability'].apply(lambda x: f"{x:.2f}")
    course_data['enjoyability'] = course_data['enjoyability'].apply(lambda x: f"{x:.2f}")
    
    return course_data.to_dict(orient='records')


update_table_data(["CS 3100"])


# In[587]:


@callback(
    Output("search_results", "data", allow_duplicate=True),
    Input('search-input', 'value'),
    prevent_initial_call=True,
)
def search_for_course(search_term):
    """
    Performs a semantic search to get the course data
    """

    search_term = str(search_term)
    
    if len(search_term) < 10:
        return [] # display nothing

    # gets the embeddings
    embedding = get_embedding(search_term)

    # searches for the embeddings
    results = index.query(vector=embedding, top_k=5)

    # gets the results
    matches = results['matches']
    ids = [match['id'] for match in matches]

    # gets the data
    data = course_reviews[course_reviews['title'].isin(ids)]
    data['instructor_rating'] = data['instructor_rating'].apply(lambda x: f"{x:.2f}")

    # returns the correct values
    return data.to_dict(orient="records")

# search_for_course("A foundational computer science course")


# In[588]:


# defining callbacks
@callback(
    Output(component_id="course_axis", component_property="figure", allow_duplicate=True),
    Input(component_id="course_dropdown", component_property="value"),
    prevent_initial_call=True,
)
def update_course_axis(courses: str):
    """
    Updates the course axis
    """
    if isinstance(courses, str): # if the course is a string
        courses = [courses]
    
    summary = pd.concat([get_course_summary_ratings(*course_components(course)) for course in courses])
    
    return course_axis(summary, course=courses)


@callback(
    Output(component_id="course_hours", component_property="figure", allow_duplicate=True),
    Input(component_id="course_dropdown", component_property="value"),
    prevent_initial_call=True,
)
def update_course_hours(courses: str):
    """
    Updates the course hours
    """
    if isinstance(courses, str):
        courses = [courses]
    
    summary = pd.concat([get_course_summary_ratings(*course_components(course)) for course in courses])

    return course_axis(summary, include="duration")


# ## Final Dashboard
# 
# Where all of the components come together to make the final dashboard. This launches the layout and the final project.

# In[589]:


# Making an app to display everything
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = Dash(
    __name__,
    title="theCourseForum Data Exploration - Data Design II: Interactive Apps",
    update_title=None,
    external_stylesheets=external_stylesheets,
)

### HEADER ###
header = html.Div(
    [
        html.Header([
        html.H1("Course Review Explorer"),
        dcc.Markdown("[GitHub](https://github.com/wkaisertexas/william-kaiser-ds-4003)")
        ], className="header"),
        html.H6("By William Kaiser for DS 4003"),
        dcc.Markdown(
            """
        The course review explorer uses data from [theCourseForum](https://thecourseforum.com), a student-run course review platform which has collected course reviews since 2006, to provide visualizations of the student experience to students and instructors. 
        
        
        Specifically, students should be able to visually compare a "shortlist" of candidates in a visual manner and search using natural (non-technical language). 
        
        Additionally, instructors should be able to understand visually what goes into a review. Misconceptions surround course reviews. Specifically, what do instructors actually care about? Seeing the relative importance of a couple of key features can show what students are keeping in mind when reviewing courses.
        """, className="introPage"
        ),
    ]
)

### BREAKING DOWN RATINGS ###
course_filter_header = html.Div(
    [
        html.H3("Breaking down ratings"),
        html.P("Compare what students said about a few courses in a visual radar plot. Good for comparing a shortlist of candidates to take in the following semester."),
    ]
)


filtering_column_names = [
    {
        "name": new_course_components_to_agg.get(key, key).replace(" ", "\n"),
        "id": key,
    }
    for i, key in enumerate(
        [
            "mnemonic",
            "course_number",
            "average",
            "instructor_rating",
            "difficulty",
            "recommendability",
            "enjoyability",
            "amount_reading",
            "amount_writing",
            "amount_group",
            "amount_homework",
        ]
    )  # new_course_components_to_agg.keys() # same thing as this, but we make mnemonic and course number first
]
course_filter_row = html.Div(
    [
        dcc.Graph(
            figure=course_axis(rows, course="CS 3100"),
            id="course_axis",
            className="one-half column",
        ),
        dcc.Graph(
            figure=course_axis(rows, course="CS 3100", include="duration"),
            id="course_hours",
            className="one-half column",
        ),
        html.Div(
            [
                dash_table.DataTable(
                    id="course_table",
                    columns=filtering_column_names,
                    data=specific_data.to_dict(orient="records"),
                    style_as_list_view=True,
                ),
            ],
            className="one-half columns",
        ),
    ],
    className="row",
)

course_filter_section = html.Div(
    [
        course_filter_header,
        course_mneumoic_dropdown,
        course_filter_row,
    ]
)

### CORRELATING INSTRUCTOR REVIEWS ###
controlling_factors = html.Div(
    [
        html.H3("Correlating Instructor Reviews"),
        html.P("Control for factors commonly cited as explaining reviews. By selecting each of these factors, the linear model trained will consider each of the following features"),
        checklist,
    ],
    className="one-third column",
)

distribution_of_reviews = html.Div(
    [
        html.H3("Distribution of Residuals"),
        html.P("What's left over once you control for these factors? Looking at the distribution of what is left over, can provide insights into student thinking (hint: look at the skew present)"),
        dcc.Graph(id="review_residuals"),
    ],
    className="one-third column",
)

predictive_power_of_features = html.Div(
    [
        html.H3("Predictive Power of Features"),
        html.P("How well do these features predict the instructor rating? What is the strength of each coefficient? Is a coefficient positive or negative?"),
        dcc.Graph(id="correlation_plot"),
    ],
    className="one-third column",
)

correlating_instructor_reviews = html.Div(
    [controlling_factors, distribution_of_reviews, predictive_power_of_features],
    className="row",
)

instructor_review_explanation = dcc.Markdown(
    """### What Makes Up a Review?

Instructor reviews have associations with GPA, Difficulty, Number of Hours Per Week and More. See how controlling for these factors changes outcomes.
"""
)

instructor_review_explanation = html.Div(
    [html.Div([instructor_review_explanation], className="one-third column")],
    className="row",
)

### SEMANTIC SEARCH ###
search_box = html.Div(
    [
        html.H2("Semantic Search"),
        html.P(
            "Describe a course you think you would like to take. Searching is semantic so there is no need to use any fancy terminology. You may just use natural language."
        ),
        dcc.Textarea(
            id="search-input",
            # type="text",  # debounce=True,
            placeholder="I want a class on computer networking...",
        ),
    ],
    className="one-third column",
    id="search-box",
)

search_table = html.Div(
    [
        html.H2("Courses Matching Your Description"),
        html.P("These are courses which an embedding model has deemed are sufficiently close to your query. Note: some of these courses are old and are not currently being taught in the next semester."),
        dash_table.DataTable(
            id="search_results",
            data=[],
            style_data={
                "whiteSpace": "normal",
                "height": "auto",
            },
            columns=[
                {
                    "name": "Course Title",
                    "id": "title",
                },
                {
                    "name": "Description",
                    "id": "description",
                },
                {"name": "Instructor Rating", "id": "instructor_rating"},
            ],
            style_cell={"textAlign": "left"},
            style_as_list_view=True,
        ),
    ],
    className="two-thirds column",
)

semantic_search = html.Div(
    [
        search_box,
        search_table,
    ],
    className="row",
)

search_section = html.Div(
    [
        html.Div(
            [
                dcc.Markdown(
                    """
## Finding a New Course

Explore courses that you may have not considered taking before, but have a proven track record of being enjoyable courses.
    """
                ),
            ],
            className="one-third column",
        )
    ],
    className="row",
)

### General Footer
footer = dcc.Markdown("""
Have any **more** thoughts about course reviews or how to make this visualization better? Please let me know! You can reach me at at [xbk6xm@virginia.edu](mailto:xbk6xm@virginia.edu). Alternatively, you could could [Submit an Issue on GitHub](https://github.com/wkaisertexas/william-kaiser-ds-4003) with your comment..
""")

### Pretty Looking Columns w/ Dashed Borders
columns = html.Div(
    [
        html.Div(),
        html.Div(),
        html.Div(),
        html.Div(),
        html.Div(),
        html.Div(),
        html.Div(),
        html.Div(),
        html.Div(),
        html.Div(),
        html.Div(),
    ],
    className="column-cont",
)

### FINAL LAYOUT ###
app.layout = html.Div(
    [
        header,
        html.Hr(),
        course_filter_section,
        html.Hr(),
        instructor_review_explanation,
        html.Hr(),
        correlating_instructor_reviews,
        html.Hr(),
        search_section,
        html.Hr(),
        semantic_search,
        html.Hr(),
        footer,
        columns,
    ],
    className="container",
)

server = app.server


# In[590]:


if __name__ == "__main__":
    # for some reason debug has not been working for me
    app.run_server(jupyter_mode="tab")

