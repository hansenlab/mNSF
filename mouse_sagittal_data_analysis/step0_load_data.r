# 
################################################################ 
################################ find HVGs
################################################################ 
dir_data="/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/data_10X_ST/mouse_Sagittal/"
# setwd(dir_data)
list.files("./")
dir_out="/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/data_10X_ST/mouse_Sagittal/put"
# dir.create(dir_out)
setwd(dir_out)


exp_Data1=Matrix::readMM(paste0(dir_data,"/V1_Mouse_Brain_Sagittal_Anterior_Section_2_filtered_feature_bc_matrix/filtered_feature_bc_matrix/",
                               "matrix.mtx" ))

exp_Data2=Matrix::readMM(paste0(dir_data,"/V1_Mouse_Brain_Sagittal_Posterior_Section_2_filtered_feature_bc_matrix/filtered_feature_bc_matrix/",
                                "matrix.mtx" ))


dim(exp_Data1)
# [1] 32285  2823
dim(exp_Data2)
# [1] 32285  2823

exp_Data1=data.matrix(exp_Data1)
exp_Data1_norm=t(t(exp_Data1)/colSums(exp_Data1))*mean(colSums(exp_Data1))
exp_Data1_norm_log=log2(exp_Data1_norm+1)

exp_Data2=data.matrix(exp_Data2)
exp_Data2_norm=t(t(exp_Data2)/colSums(exp_Data2))*mean(colSums(exp_Data2))
exp_Data2_norm_log=log2(exp_Data2_norm+1)

library(matrixStats)
rowVars1=rowVars(exp_Data1_norm_log)
rowVars2=rowVars(exp_Data2_norm_log)


rowVars12=rowVars1+rowVars2

which_hvg=order(rowVars12,decreasing = T)[1:500]
saveRDS(which_hvg,file="which_hvg.rds")

which_hvg=readRDS("which_hvg.rds")



################################################################################################################################ 
####################################################### process sample 1
################################################################################################################################ 

# data downloaded from 10X website:https://www.10xgenomics.com/resources/datasets?menu%5Bproducts.name%5D=Spatial%20Gene%20Expression&query=&page=1&configure%5Bfacets%5D%5B0%5D=chemistryVersionAndThroughput&configure%5Bfacets%5D%5B1%5D=pipeline.version&configure%5BhitsPerPage%5D=500&configure%5BmaxValuesPerFacet%5D=1000
# Spatial Gene Expression Dataset by Space Ranger 2.0.0
# visium demonstration, v1 chemistry

# dir.create(dir_out)
exp_Data=Matrix::readMM(paste0(dir_data,"/V1_Mouse_Brain_Sagittal_Anterior_Section_2_filtered_feature_bc_matrix/filtered_feature_bc_matrix/",
                              "matrix.mtx" ))
# setwd(dir_out)
exp_Data=data.matrix(exp_Data)
Y=t(exp_Data[which_hvg,])


write.csv(Y,row.names = F,#col.names = F,
          file=paste0("Y_sample1.csv"))



#### load barcode info
barcode_=read.delim(paste0(dir_data,"/V1_Mouse_Brain_Sagittal_Anterior_Section_2_filtered_feature_bc_matrix/filtered_feature_bc_matrix/",
                           "/barcodes.tsv.gz"),header = F)
barcode_=barcode_$V1
# head(barcode_)
length(barcode_)
# 2823

#### load coordinate info for each barcode
pos_=read.csv(paste0(dir_data,"/V1_Mouse_Brain_Sagittal_Anterior_Section_2_spatial/",
                     "spatial/tissue_positions.csv"),header = F)

colnames(pos_)=c("barcode","if_inTissue","pos_x","pos_y","pos_x_image","pos_y_image")

pos_filt=pos_[pos_$barcode %in% barcode_,]
rownames(pos_filt)=pos_filt$barcode
pos_filt=pos_filt[barcode_,]
ls(pos_filt)


pos_x=pos_filt$pos_x_image
pos_y=pos_filt$pos_y_image

X=data.frame(x=pos_x,y=pos_y)

write.csv(X,row.names = F,#col.names = F,
          file=paste0("X_sample1.csv"))


#### 


################################################################################################################################ 
####################################################### process sample 2
################################################################################################################################ 

# data downloaded from 10X website:https://www.10xgenomics.com/resources/datasets?menu%5Bproducts.name%5D=Spatial%20Gene%20Expression&query=&page=1&configure%5Bfacets%5D%5B0%5D=chemistryVersionAndThroughput&configure%5Bfacets%5D%5B1%5D=pipeline.version&configure%5BhitsPerPage%5D=500&configure%5BmaxValuesPerFacet%5D=1000
# Spatial Gene Expression Dataset by Space Ranger 2.0.0
# visium demonstration, v1 chemistry

# dir.create(dir_out)
exp_Data=Matrix::readMM(paste0(dir_data,"/V1_Mouse_Brain_Sagittal_Posterior_Section_2_filtered_feature_bc_matrix/filtered_feature_bc_matrix/",
                               "matrix.mtx" ))
# setwd(dir_out)
exp_Data=data.matrix(exp_Data)
Y=t(exp_Data[which_hvg,])


write.csv(Y,row.names = F,#col.names = F,
          file=paste0("Y_sample2.csv"))



#### load barcode info
barcode_=read.delim(paste0(dir_data,"/V1_Mouse_Brain_Sagittal_Posterior_Section_2_filtered_feature_bc_matrix/filtered_feature_bc_matrix/",
                           "/barcodes.tsv.gz"),header = F)
barcode_=barcode_$V1
# head(barcode_)
length(barcode_)
# 2823

#### load coordinate info for each barcode
pos_=read.csv(paste0(dir_data,"/V1_Mouse_Brain_Sagittal_Posterior_Section_2_spatial/",
                     "spatial/tissue_positions.csv"),header = F)

colnames(pos_)=c("barcode","if_inTissue","pos_x","pos_y","pos_x_image","pos_y_image")

pos_filt=pos_[pos_$barcode %in% barcode_,]
rownames(pos_filt)=pos_filt$barcode
pos_filt=pos_filt[barcode_,]
ls(pos_filt)


pos_x=pos_filt$pos_x_image
pos_y=pos_filt$pos_y_image

X=data.frame(x=pos_x,y=pos_y)

write.csv(X,row.names = F,#col.names = F,
          file=paste0("X_sample2.csv"))


#### 






