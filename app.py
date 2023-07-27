from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import mlem
import toml
import io


# implemento la funzione main
def main():
    new_model = mlem.api.load('model_.mlem')
    st.header("Input values")
    
    AdultMortality= st.number_input('Enter a NUMBER value Adult Mortality:',value=1,step=1)
    Measles = st.number_input('Enter a NUMBER value Measles:', value=0, step=1)
    IncomeCompOfResources = st.number_input('Enter a NUMBER value Income Composition of Resources:', value=0.3, step=0.01)
    Schooling = st.number_input('Enter a NUMBER value Schooling:', value=4.9, step=0.1)
    if st.button('Calculate Y_PRED'):
        st.write("Y_PRED: ",round(new_model.predict([[ Schooling,IncomeCompOfResources,Measles,AdultMortality]])[0],4))
    
    
    
    


# questo modulo sarà eseguito solo se runnato
if __name__ == "__main__":
    main()