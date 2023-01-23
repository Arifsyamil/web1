#Import all neccessary libraries
import streamlit as st
import re
import wikipediaapi
import malaya
import torch
import tensorflow
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import tracemalloc

#Page header, title
st.set_page_config(page_title= "Malay Named Entity Recognition (NER) Model", page_icon= ":book:", layout= "wide")
st.title(":book: Malay Named Entity Recognition (NER) model")
st.markdown("Sila tekan butang di bawah untuk mulakan program")
btn_main = st.button("TEKAN MULA")
if btn_main:
	st.write("BERJAYA")
else:
	st.write("TIDAK TEKAN")

@st.cache
def app():
	return None


#CREATE TEXT FORM
with st.form(key= 'my_form'):
	kata = st.text_area(label="Sila taip teks atau ayat:", max_chars= 500)
	
	btn_model = st.radio("Pilih model untuk pengecaman entiti nama",
		("KNN", "BERT", "Tiny-BERT", "ALBERT", "Tiny-ALBERT", "XLNET", "ALXLNET", "FASTFORMER", "Tiny-FASTFORMER"))
	
	if btn_model == 'KNN':
		st.write('Anda pilih model KNN.')
	elif btn_model == 'BERT':
		st.write('Anda pilih model BERT')
	elif btn_model == 'Tiny-BERT':
		st.write('Anda pilih model Tiny-BERT')
	elif btn_model == 'ALBERT':
		st.write('Anda pilih model ALBERT')
	elif btn_model == 'Tiny-ALBERT':
		st.write('Anda pilih model Tiny-ALBERT')
	elif btn_model == 'XLNET':
		st.write('Anda pilih model XLNET')
	elif btn_model == 'ALXLNET':
		st.write('Anda pilih model ALXLNET')
	elif btn_model == 'FASTFORMER':
		st.write('Anda pilih model FASTFORMER')
	elif btn_model == 'Tiny-FASTFORMER':
		st.write('Anda pilih model Tiny-FASTFORMER')												
	else:
		st.write("Sila pilih satu model untuk pengecaman entiti nama")

	submit_button = st.form_submit_button(label= ":arrow_right: Buat Ramalan")

if submit_button:
    if re.sub(r'\s+','',kata)=='':
        st.error('Ruangan teks tidak boleh kosong.')

    elif re.match(r'\A\s*\w+\s*\Z', kata):
        st.error("Teks atau ayat mestilah sekurang-kurangnya dua patah perkataan.")
    
    else:
    	st.markdown("### Hasil Ramalan")
    	#result1 = find_text()
    	#result2 = clean_data()
    	#result3 = use_malaya()
    	#result4 = data_model()
    	#result5 = train_model()
    	#result6 = evaluate_model()
    	#result7 = hypertune_model()
    	#result8 = o_v_model()
    	#result9 = error_rate()
    	#result10 = ramal_kata()

    	st.success("Butang hantar berfungsi!")

#About model
with st.expander("About this app", expanded=True):
    st.write(
        """     
-   **Pengecaman Nama Entiti Malay** adalah sebuah aplikasi pembelajaran mesin yang dibangunkan bagi mengecam entiti pada setiap token menggunankan modul MALAYA
-   Entiti yang dikaji ialah: LOKASI, MANUSIA, ORGANISASI 
-   Aplikasi ini menggunakan BERT yang mempunyai accuracy score yang paling tinggi dalam modul MALAYA
-   Model ini mempunyai 3 fitur utama iaitu kata, kata sebelum dan kata selepas. Kelas yang disasarkan adalah LOKASI, MANUSIA dan ORGANISASI
-   Maklumat lanjut boleh hubungi Muhd Arif Syamil melalui emel a177313@siswa.ukm.edu.my atau 012-7049021 
       """
    )
