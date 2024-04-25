# William Kaiser's DS4003 Term Project


- [Sprint 2 - Data](./sprints/sprint2-data.ipynb)
- [Sprint Render Deploy (Initial Test)](https://william-kaiser-ds-4003.onrender.com/)

## Description

This is a term project for DS 4003 - Data Design II at the University of Virginia. In this dashboard, course data from the University of Virginia was analyzed from [theCourseFourm.com](https://thecourseforum.com) which is a student run review site for courses at the University of Virginia. The data was collected from the site and analyzed from two perspectives: the course and the professor. 

Students primarily seek to: compare courses and find new courses. This dashboard start with the most common use case of students: comparing shortlists of courses and finding new course. 

From the instructor's perspective, instructors have many misconceptions about how students perceive their courses and what students care about when it comes to reviews. A few features were added 

## Building Process

Data was collected through a direct data connection to [theCourseForum.com](https://thecorusefourm.com) as I am a member and have permissions access to the site through my role as a student maintainer. 

This dashboard was building [Dash](https://dash.plotly.com) and deployed on [Render](https://render.com). The dashboard was built in a Jupyter notebook and then converted into `app.py` for deployment. 

A vector database and an embedding model were used in conjunction to provide a semantic search feature for the dashboard. [Pinecone](https://pinecone.io) was used with OpenAi's GPT model to provide a semantic search feature for the dashboard. Both choices were largely made out of the limitations of render and the limited CPU and memory resources available.

## Strengths / Weaknesses and Learning Experiences

The strengths of this project is: the technology stack used -- dash is a very powerful tool for building dashboards. However, the project required a static `data.csv` for analysis. Given this, the data is static and not updated in real time.

Learning how to use both an embedding model (OpenAI's GPT model) and a vector search database (Pinecone) was a valuable learning experience and something that I had never done before. I think I ended up spending too much time trying to figure this out and this time would have been better spend elsewhere. 

