
import pandas as pd
import ast

print("Modules imported successfully")

#Reading the datasets
movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')
credits.rename(columns={'movie_id':'id'},inplace=True)
print("Datasets read successfully")

#Merging the datasets on 'id' column
movies=movies.merge(credits,on='id')
print("Datasets merged successfully")

#Selecting relevant columns
movies=movies[['id','original_title','overview','genres','keywords','cast','crew','release_date','vote_average']]
print("Selected relevant columns")

def convert(text):
    L=[]
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

def convert3(text):
    L=[]
    counter=0
    for i in ast.literal_eval(text):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break   
    return L

def fetch_director(text):
    L=[]    
    for i in ast.literal_eval(text):
        if i['job']=='Director':
            L.append(i['name'])
    return L

#Dropping rows with missing values in relevant columns
movies.dropna(inplace=True)
print("Dropped rows with missing values")


#Applying the functions to relevant columns
movies['genres']=movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert)
movies['cast']=movies['cast'].apply(convert3)
movies['crew']=movies['crew'].apply(fetch_director)

#Saving a copy of original overview before cleaning
movies['overview_raw']=movies['overview']
movies['overview']=movies['overview'].apply(lambda x:x.split())

#Removing spaces in names within lists
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

#Creating a new column 'tags' by combining relevant columns and converting to lowercase and space-separated string
movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


#Final dataframe to be used for model building
new_df=movies[['id','original_title','overview_raw','tags','release_date','vote_average']]
new_df.rename(columns={'overview_raw':'overview'},inplace=True)

#Converting list of tags to space-separated lowercase string
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x).lower())


#Saving the cleaned data to a new CSV file
print("Saving cleaned data to 'cleaned_movies_data.csv'...")
new_df.to_csv('cleaned_movies_data.csv',index=False)
print("Data cleaning completed and saved to 'cleaned_movies_data.csv'")
