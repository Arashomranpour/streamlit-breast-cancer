import pickle 
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go




st.set_page_config(page_title="Breast Cancer ML predictor"
                   ,page_icon=":female-doctor:",
                   )

with open("./style.css") as f:
    st.markdown('<style>' + f.read() + '</style>', unsafe_allow_html=True)
def get_data():
    data=pd.read_csv("./data.csv")
    # print(data.head())
    data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
    data["diagnosis"]=data["diagnosis"].map({"M":1,"B":0})
    return data


data=get_data()

labels=[x for x in data.columns]
tuple_labels = [(item, item) for item in labels]
tuple_labels.remove(("diagnosis","diagnosis"))
# st.text(tuple_labels)

input_dict={}
with st.sidebar:
    st.sidebar.header("Measurements")
    for label ,key in tuple_labels:
        input_dict[key]=st.sidebar.slider(
            label=label,
            min_value=0.0,
            max_value=float(data[key].max()),
            value=float(data[key].mean())
            
            
            )

def scaler(input_dict):
    data=get_data()
    x=data.drop("diagnosis",axis=1)
    scaled={}
    for key,value in input_dict.items():
        max_value=x[key].max()
        min_value=x[key].min()
        scaled_value=(value-min_value)/(max_value-min_value)
        scaled[key]=scaled_value
    return scaled

def get_chart(input_dict):
    input_dict=scaler(input_dict)
    categories = ["Raduis","Texture","Perimeter","Area","Smoothness","Compactness","Concavity","Concave Points","Symetry","Fractal Dimensions"]
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
      r=[
          input_dict["radius_mean"],
          input_dict["texture_mean"],
          input_dict["perimeter_mean"],
          input_dict["area_mean"],
          input_dict["smoothness_mean"],
          input_dict["compactness_mean"],
          input_dict["concavity_mean"],
          input_dict["concave points_mean"],
          input_dict["symmetry_mean"],
          input_dict["fractal_dimension_mean"]
          
          ],
      theta=categories,
      fill='toself',
      name='Mean Value'
))
    fig.add_trace(go.Scatterpolar(
      r=[input_dict["radius_se"],
         input_dict["texture_se"],
         input_dict["perimeter_se"],
         input_dict["area_se"],
         input_dict["smoothness_se"],
         input_dict["compactness_se"],
         input_dict["concavity_se"],
         input_dict["concave points_se"],
         input_dict["symmetry_se"],
         input_dict["fractal_dimension_se"],
          
          ],
      theta=categories,
      fill='toself',
      name='Standard Error'
))
    
    fig.add_trace(go.Scatterpolar(
      r=[input_dict["radius_worst"],
         input_dict["texture_worst"],
         input_dict["perimeter_worst"],
         input_dict["area_worst"],
         input_dict["smoothness_worst"],
         input_dict["compactness_worst"],
         input_dict["concavity_worst"],
         input_dict["concave points_worst"],
         input_dict["symmetry_worst"],
         input_dict["fractal_dimension_worst"],
          
          ],
      theta=categories,
      fill='toself',
      name='Worst Value'
))

    fig.update_layout(
        width=500,
        height=700,
        
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 1]
      
    )),
  showlegend=True
)
    
    
    
    return fig


def model_pre(input):
    model=pickle.load(open("./model.pkl","rb"))
    scaler=pickle.load(open("./scaler.pkl","rb"))
    
    input_array=np.array(list(input.values())).reshape(1,-1)
    # st.dataframe(input_array)
    scld=scaler.transform(input_array)
    prediction=model.predict(scld)
    
    st.subheader("Cell Cluster prediction")
    st.write("Cell Cluster is")
    
    if prediction[0]==0:
        st.write("<span class='diagnosis benign'>Benign</span>",unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>",unsafe_allow_html=True)
        
    st.write("Probability of being Benign :",model.predict_proba(scld)[0][0])
    st.write("Probability of being Malicious :",model.predict_proba(scld)[0][1])
    st.write("This app can assist medial professionals in making diagnosis , do not use as substitue for professional diagnosis.")


# col1,col2=st.columns([5,2])
# with col1:    
with st.container():
    st.title("Breast Cancer Predictor")
    st.write("Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample")
    
    col1,col2=st.columns([3,1])
    # col1 4 barabar col2
    
    with col1:
        
        chart=get_chart(input_dict)
    
        st.plotly_chart(chart)
    
        
        
    with col2:
       
        model_pre(input_dict)
        
        # st.write(f"<span class='col2'>{t}</span>",unsafe_allow_html=True)
        
