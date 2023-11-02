## Load the package
library('SingleCellExperiment')
library('sparseMatrixStats')
library(patchwork)
library(dplyr)
library(matrixStats)



#############################################################
#############################################################
#############################################################
dir_out=""
setwd(dir_out)

dir_processedData = ""
setwd(dir_processedData)

#############################################################
## load the spot-level data
# load("/users/ywang/Hansen_projects/scRNA/data/sce_.RData")
# save(sce,file="/home-net/home-3/mdabrow1@jhu.edu/scratch/spatial/data/sce_.RData")#sce, on marcc
a=colData(sce)
d <- as.data.frame(colData(sce))
b=counts(sce)
genes_all=rownames(b)

sample_name_uni=unique(d$sample_name)

metadata_per_rep=d[,c("subject","position","sample_name","replicate","sum_umi")]
metadata_per_rep=metadata_per_rep[!duplicated(metadata_per_rep$sample_name),]
metadata_per_rep$subject_position=paste0(metadata_per_rep$subject,"_",metadata_per_rep$position)


save(metadata_per_rep,file="metadata_per_rep.RData")
save(sample_name_uni,file="sample_name_uni.RData")
save(genes_all,file="/genes_all.RData")





################################################################################################
#### load counts data
################################################################################################

mat_rowMedians=array(dim=c(length(genes_all),12))
for(j in 1:12){
  print(j)
  
  sample_name_tmp_=sample_name_uni[j]
  exp_Data=b[,d$sample_name==sample_name_tmp_]

  rownames(exp_Data)=genes_all

  exp_Data=(data.matrix(exp_Data[,]))
  
  mat_rowMedians[,j]=rowMeans(exp_Data==0)
  
}

mat_rowMedians_min=rowMins(mat_rowMedians)# the lowest sparsity level among all the samples
which_notSparse=which(mat_rowMedians_min<0.98)
saveRDS(which_notSparse,
        file=paste0(dir_processedData,"which_notSparse_2percent.rds"))

saveRDS(mat_rowMedians_min,
        file=paste0(dir_processedData,"mat_rowMedians_min.rds"))

length(which_notSparse)
# 13122

# > dim(exp_Data)
# [1] 33538  3460


################################################################################################
which_notSparse = readRDS(paste0(dir_processedData,"which_notSparse_2percent.rds"))
for( j in 12){
  print(j)
  
  sample_name_tmp_=sample_name_uni[j]
  counts_sample_tmp_=b[, d$sample_name==sample_name_tmp_]
  print(j)
  
  counts_sample_tmp_ = t(data.matrix(counts_sample_tmp_[which_notSparse, ]))
  write.csv(counts_sample_tmp_,row.names = F,
            file=paste0(dir_processedData,"Y_alllGenes_2percentSparseFilter_sample",j,".csv"))
  
  print(dim(counts_sample_tmp_))

}
######################################################### process position data
### get position info for each sample 
for(j in 1:12){
  print(j)
  sample_name_tmp_=sample_name_uni[j]
  pos_col_sample_tmp_=d$imagecol[d$sample_name==sample_name_tmp_]
  
  pos_row_sample_tmp_=d$imagerow[d$sample_name==sample_name_tmp_]
  save(pos_col_sample_tmp_,pos_row_sample_tmp_,
       file=paste0("pos_row_col_sample_tmp_Sample",j,".RData"))
}


for(j in 1:12){
  print(j)
  load(paste0("pos_row_col_sample_tmp_Sample",j,".RData"))
  # print(length(pos_col_sample_tmp_))
  X_allSpots=cbind(pos_col_sample_tmp_,pos_row_sample_tmp_)[,]
  write.csv(X_allSpots,
            file=paste0("X_allSpots_sample",j,".csv"),row.names = F)
  
}



########################################################################
########################################################################
##########after run the python code for calculating poisson deviance
########################################################################

var_mat=array(dim=c(length(which_notSparse),12))
for(ksample in 1:12){

  print(ksample)
  
  var_=read.csv(paste0("dev_poisson_sample",ksample,".csv"))
  rownames(var_)=var_[,1]
  
  var_mat[,ksample]=var_[genes_all[which_notSparse],2]
}
var_mat_rowMaxs=rowMaxs(var_mat)
names(var_mat_rowMaxs)=genes_all[which_notSparse]

saveRDS(var_mat_rowMaxs,file="var_mat_rowMaxs.rds")

features_ordered=names(var_mat_rowMaxs)[order(var_mat_rowMaxs,decreasing = T)]
features_sele=features_ordered[1:2000]
saveRDS(features_ordered,file="features_ordered.rds")
saveRDS(features_sele,file="features_sele.rds")



##############################
features_sele=readRDS("features_sele.rds")
for(ksample in 1:12){
  # https://satijalab.org/seurat/articles/spatial_vignette.html
  print(ksample)
  
  sample_name_tmp_=sample_name_uni[ksample]
  exp_Data=b[, d$sample_name==sample_name_tmp_]

  Y=t(data.matrix(exp_Data[features_sele,]))

  write.csv(Y,row.names = F,#col.names = F,
            file=paste0("Y_features_sele_sample",ksample,"_2000genes.csv"))
  
  Y=t(data.matrix(exp_Data[features_sele[1:500],]))
  
  write.csv(Y,row.names = F,#col.names = F,
            file=paste0("Y_features_sele_sample",ksample,"_500genes.csv"))
}





