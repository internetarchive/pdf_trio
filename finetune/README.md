Training (fine tuning) data as a list of URLs. Each PDF used in fine tuning is here. 
** Negative: 0_urls.tsv
** Positive: 1_urls.tsv
** Positive: 2_urls.tsv

Notice that there are two files for positive examples. The 2_urls.tsv file contains PDFs associated with "long tail". All the positive cases are used the same way during fine tuning, they are separate here to show the provenance. To recreate the PDF training set, fetch the URLs using curl or wget, make sure the output is a PDF and not a 404 error page, collecting the PDFs in separate directories. The positives can be all put into one dir. For example, make 2 directories named '0' and '1' and put the PDFs in there. 

