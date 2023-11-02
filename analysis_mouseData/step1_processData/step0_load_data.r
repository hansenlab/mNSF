library(matrixStats)



################################################################ 
################################ find HVGs
################################################################ 
dir_data="/dcs04/hansen/data/ywang/ST/data_10X_ST//mouse_Sagittal/"
# setwd(dir_data)
list.files("./")
dir_out="/dcs04/hansen/data/ywang/ST/data_10X_ST//mouse_Sagittal/put"
# dir.create(dir_out)
setwd(dir_out)


################################################################################################

list_subdir=list()
list_subdir[[1]]="V1_Mouse_Brain_Sagittal_Anterior_filtered_feature_bc_matrix"
list_subdir[[2]]="V1_Mouse_Brain_Sagittal_Anterior_Section_2_filtered_feature_bc_matrix"
list_subdir[[3]]="V1_Mouse_Brain_Sagittal_Posterior_filtered_feature_bc_matrix"
list_subdir[[4]]="V1_Mouse_Brain_Sagittal_Posterior_Section_2_filtered_feature_bc_matrix"

list_subdirSp=list()
list_subdirSp[[1]]="V1_Mouse_Brain_Sagittal_Anterior_spatial"
list_subdirSp[[2]]="V1_Mouse_Brain_Sagittal_Anterior_Section_2_spatial"
list_subdirSp[[3]]="V1_Mouse_Brain_Sagittal_Posterior_spatial"
list_subdirSp[[4]]="V1_Mouse_Brain_Sagittal_Posterior_Section_2_spatial"

################################################################################################
############# save the Y counts
################################################################################################
dir_data="/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/"
ksample=1
library(Seurat)




################################################################################################
#### load feature info
################################################################################################
dir_data="/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/"

features=read.delim(paste0(dir_data,"/V1_Mouse_Brain_Sagittal_Posterior_Section_2_filtered_feature_bc_matrix/filtered_feature_bc_matrix/",
                           "/features.tsv.gz"),header = F)

# list.files(paste0(dir_data,"/V1_Mouse_Brain_Sagittal_Posterior_Section_2_filtered_feature_bc_matrix/filtered_feature_bc_matrix/"))
ls(features)
features=features$V2


################################################################################################
#### load counts data
################################################################################################

# concatenate the counts
# list_rowMedians=list()
mat_rowMedians=array(dim=c(length(features),4))
# list_counts_perSample=list()
for(ksample in 1:4){
  # https://satijalab.org/seurat/articles/spatial_vignette.html
  print(ksample)
  dir_data="/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/"
  
  exp_Data=Matrix::readMM(paste0(dir_data,list_subdir[[ksample]],"/filtered_feature_bc_matrix/",
                                 "matrix.mtx.gz" ))
  print(dim(exp_Data))
  
  colnames(exp_Data)=as.character(1:ncol(exp_Data))
  rownames(exp_Data)=features
  # obj=CreateSeuratObject(exp_Data)
  # obj=FindVariableFeatures(obj,selection.method = "dispersion",nfeatures=2000)
  Y=(data.matrix(exp_Data[,]))

  mat_rowMedians[,ksample]=rowMeans(Y==0)

}


mat_rowMedians_min=rowMins(mat_rowMedians)
which_notSparse=which(mat_rowMedians_min<0.98)
saveRDS(which_notSparse,file="which_notSparse_2percent.rds")
saveRDS(mat_rowMedians_min,file="mat_rowMedians_min.rds")



for(ksample in 1:4){
  # https://satijalab.org/seurat/articles/spatial_vignette.html
  print(ksample)
  dir_data="/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/"
  
  exp_Data=Matrix::readMM(paste0(dir_data,list_subdir[[ksample]],"/filtered_feature_bc_matrix/",
                                 "matrix.mtx.gz" ))
  
  colnames(exp_Data)=as.character(1:ncol(exp_Data))
  rownames(exp_Data)=features

  Y=t(data.matrix(exp_Data[mat_rowMedians_min<0.98,]))

  write.csv(Y,row.names = F,#col.names = F,
            file=paste0("Y_alllGenes_2percentSparseFilter_sample",ksample,"_v2.csv"))
  
}





##################################################################
##################################################################
############## after run the python code for calculating poisson deviance
##################################################################
##################################################################
var_mat=array(dim=c(sum(mat_rowMedians_min<0.98),4))
for(ksample in 1:4){

  
  var_=read.csv(paste0("/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal/put/dev_poisson_sample",ksample,".csv"))
  rownames(var_)=var_[,1]
  
  var_mat[,ksample]=var_[features[mat_rowMedians_min<0.98],2]
}
var_mat_rowMaxs=rowMaxs(var_mat)
names(var_mat_rowMaxs)=features[mat_rowMedians_min<0.98]


saveRDS(var_mat_rowMaxs,file="var_mat_rowMaxs.rds")

features_ordered=names(var_mat_rowMaxs)[order(var_mat_rowMaxs,decreasing = T)]
features_sele=features_ordered[1:2000]
saveRDS(features_ordered,file="features_ordered.rds")
saveRDS(features_sele,file="features_sele.rds")



features_sele=readRDS("features_sele.rds")
######################################################33
######################################################
for(ksample in 1:4){
  # https://satijalab.org/seurat/articles/spatial_vignette.html
  print(ksample)
  dir_data="/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/"
  
  exp_Data=Matrix::readMM(paste0(dir_data,list_subdir[[ksample]],"/filtered_feature_bc_matrix/",
                                 "matrix.mtx.gz" ))
  
  colnames(exp_Data)=as.character(1:ncol(exp_Data))
  rownames(exp_Data)=features

  Y=t(data.matrix(exp_Data[features_sele,]))
  # 
  write.csv(Y,row.names = F,#col.names = F,
            file=paste0("Y_features_sele_sample",ksample,"_v2.csv"))
  
  Y=t(data.matrix(exp_Data[features_sele[1:500],]))
  
  write.csv(Y,row.names = F,#col.names = F,
            file=paste0("Y_features_sele_sample",ksample,"_v2_500genes.csv"))
}




