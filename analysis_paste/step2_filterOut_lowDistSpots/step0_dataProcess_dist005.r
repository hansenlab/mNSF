

### load pacakges 
library(tidyverse)
library(ggplot2)
library(Matrix)
library(ggforce)#
library(cowplot)
library(RColorBrewer)
library(grid)
### 

### 
# tech reps
list_samplePairs=list()
list_samplePairs[[1]]=c(1,2)
list_samplePairs[[2]]=c(3,4)
list_samplePairs[[3]]=c(5,6)
list_samplePairs[[4]]=c(7,8)
list_samplePairs[[5]]=c(9,10)
list_samplePairs[[6]]=c(11,12)

# bio reps
list_samplePairs[[7]]=c(2,3)
list_samplePairs[[8]]=c(6,7)
list_samplePairs[[9]]=c(10,11)

####################### 
dir_processedData=""
setwd(dir_processedData)
dir_processedData_dist005_filtered = ""
dir_fullData = ""
dir_pasteData = ""


####################### get Y_kp
for(sample_ in c(1:6)){
  X_allSpots=read.csv(paste0(dir_fullData, "//X_allSpots_sample",sample_,".csv"))
  X_allSpots_paste_ori_cor1=read.csv(paste0(dir_pasteData,"/coord_ori_X_s",sample_,".csv"))
  X_allSpots_paste_ori_cor2=read.csv(paste0(dir_pasteData,"/coord_ori_Y_s",sample_,".csv"))

  X_allSpots_paste_adj_cor1=read.csv(paste0(dir_pasteData,"coord_paste_X_s",sample_,".csv"))
  X_allSpots_paste_adj_cor2=read.csv(paste0(dir_pasteData,"coord_paste_Y_s",sample_,".csv"))

  # load Y counts, with selected features
  Y=read.csv(paste0("Y_features_sele_sample",sample_,"_500genes.csv"))

  rownames(Y)=paste0(round(X_allSpots[,1]),"_",round(X_allSpots[,2]))
  Y_kp=Y[paste0(round(X_allSpots_paste_ori_cor1[,2]),"_",
                round(X_allSpots_paste_ori_cor2[,2])),]

  # save data
  write.csv(Y_kp,file=paste0("Y_features_sele_sample",
                             sample_,"_kp_500genes.csv"),
            row.names = F)


  saveRDS(Y_kp,file=paste0("Y_features_sele_sample",
                           sample_,"_kp_500genes.rds"))

  print(sample_)
  print(dim(Y_kp))
}



########################
########################
kpair=2

for(kpair in 1:9){
    
  print(kpair)
  sample_s1=list_samplePairs[[kpair]][1]
  sample_s2=list_samplePairs[[kpair]][2]
  

  #############################################################################################
  #############################################################################################
  X_allSpots_paste_adj_s1_cor1=read.csv(paste0(dir_pasteData,"coord_paste_X_s",sample_s1,".csv"))
  X_allSpots_paste_adj_s2_cor1=read.csv(paste0(dir_pasteData,"coord_paste_X_s",sample_s2,".csv"))
  X_allSpots_paste_adj_s1_cor2=read.csv(paste0(dir_pasteData,"coord_paste_Y_s",sample_s1,".csv"))
  X_allSpots_paste_adj_s2_cor2=read.csv(paste0(dir_pasteData,"coord_paste_Y_s",sample_s2,".csv"))
  
  
  X_allSpots_paste_adj_s1_cor12=cbind(X_allSpots_paste_adj_s1_cor2[,2],
                                       X_allSpots_paste_adj_s1_cor1[,2])
  X_allSpots_paste_adj_s2_cor12=cbind(X_allSpots_paste_adj_s2_cor2[,2],
                                       X_allSpots_paste_adj_s2_cor1[,2])
  
  X_allSpots_paste_adj_concatenated_s1_s2=rbind(X_allSpots_paste_adj_s1_cor12,
                                                  X_allSpots_paste_adj_s2_cor12)
  
  
  n1=nrow(X_allSpots_paste_adj_s1_cor12)
  n2=nrow(X_allSpots_paste_adj_s2_cor12)
  
  Y_kp_s1=read.csv(paste0("Y_features_sele_sample",sample_s1,"_kp_500genes.csv"))
  Y_kp_s2=read.csv(paste0("Y_features_sele_sample",sample_s2,"_kp_500genes.csv"))
  Y_kp_s1_s2=rbind(Y_kp_s1,Y_kp_s2)

  ### cal dist
  dist_ = dist(X_allSpots_paste_adj_concatenated_s1_s2)
  dist_=data.matrix(dist_)
  diag(dist_)=1
  print(min(dist_))
  which_dist_near = which(dist_<0.1, arr.ind = T)[,1]
  length(which_dist_near)
  print(summary(as.numeric(dist_)))
  
  
  ### filter short-dist spots

  
  if(length(which_dist_near)>0){
    which_dist_near = which_dist_near[1:(length(which_dist_near)/2)]
    
    
    
    which_s1 = which_dist_near[which_dist_near<=n1]
    which_s2 = which_dist_near[which_dist_near>n1] - n1
      
    X_allSpots_paste_adj_concatenated_s1_s2 = X_allSpots_paste_adj_concatenated_s1_s2[-which_dist_near,]
    Y_kp_s1_s2 = Y_kp_s1_s2[-which_dist_near,]
    
    if(length(which_s1)>0){
      Y_kp_s1 = Y_kp_s1[-which_s1,]
    }
    if(length(which_s2)>0){
      Y_kp_s2 = Y_kp_s2[-which_s2,]
    }
    
    
  }else{
    which_s1=which_s2=which_dist_near

  }
  
  
  
  saveRDS(X_allSpots_paste_adj_concatenated_s1_s2,
          file=paste0(dir_processedData,"X_allSpots_paste_adj_concatenated_s",sample_s1,"_s",sample_s2,
                      "_filterDist005.rds"))
  
  saveRDS(Y_kp_s1_s2,
          file=paste0(dir_processedData,"Y_kp_s",sample_s1,"_s",sample_s2,##!!!
                      "_filterDist005.rds"))
  write.csv(Y_kp_s1_s2,row.names = F,#col.names = F,
            file=paste0(dir_processedData,"Y_kp_s",sample_s1,"_s",sample_s2,
                        "_filterDist005.csv"))
  write.csv(X_allSpots_paste_adj_concatenated_s1_s2,row.names = F,#col.names = F,
            file=paste0(dir_processedData,"X_adj_paste_kp_s",sample_s1,"_s",sample_s2,
                        "_filterDist005.csv"))

  write.csv(Y_kp_s1, row.names = F,#col.names = F,
            file=paste0(dir_processedData,"Y_kp_s",sample_s1,"_s",sample_s2,
                        "_filterDist005_s1_corrected.csv"))
  write.csv(Y_kp_s2, row.names = F,#col.names = F,
            file=paste0(dir_processedData,"Y_kp_s",sample_s1,"_s",sample_s2,
                        "_filterDist005_s2_corrected.csv"))
  
  dim(X_allSpots_paste_adj_concatenated_s1_s2)
  # [1] 8796    2
  # /dcs04/hansen/data/ywang/ST/DLPFC/processed_Data_dist005/Y_kp_s3_s4_filterDist005_corrected.csv
  ##############################################################################################
  ##############################################################################################
  X_allSpots_paste_ori_cor1_s1=read.csv(paste0(dir_pasteData,"/coord_ori_X_s",sample_s1,".csv"))
  X_allSpots_paste_ori_cor2_s1=read.csv(paste0(dir_pasteData,"/coord_ori_Y_s",sample_s1,".csv"))
  X_allSpots_paste_ori_cor1_s2=read.csv(paste0(dir_pasteData,"/coord_ori_X_s",sample_s2,".csv"))
  X_allSpots_paste_ori_cor2_s2=read.csv(paste0(dir_pasteData,"/coord_ori_Y_s",sample_s2,".csv"))
  
  X_ori_kpSpots_s1 = cbind(X_allSpots_paste_ori_cor2_s1[,2],
                           X_allSpots_paste_ori_cor1_s1[,2])
  X_ori_kpSpots_s2 = cbind(X_allSpots_paste_ori_cor2_s2[,2],
                            X_allSpots_paste_ori_cor1_s2[,2])
  

  ### 

  
  if(length(which_s1)>0){
    X_ori_kpSpots_s1 = X_ori_kpSpots_s1[-which_s1,]
  }
  
  if(length(which_s2)>0){
    X_ori_kpSpots_s2 = X_ori_kpSpots_s2[-which_s2,]
  }
  
  write.csv(X_ori_kpSpots_s1, row.names = F,#col.names = F,
            file=paste0(dir_processedData,"X_kp_s",sample_s1,"_s",sample_s2,
                        "_filterDist005_s1_corrected.csv"))
  write.csv(X_ori_kpSpots_s2, row.names = F,#col.names = F,
            file=paste0(dir_processedData,"X_kp_s",sample_s1,"_s",sample_s2,
                        "_filterDist005_s2_corrected.csv"))
  
  ###### 
  rownames_kpSpots_s1 = paste0(round(X_ori_kpSpots_s1[,1],2),"_",round(X_ori_kpSpots_s1[,2],2))
  rownames_kpSpots_s2 = paste0(round(X_ori_kpSpots_s2[,1],2),"_",round(X_ori_kpSpots_s2[,2],2))
    
  rownames_kpSpots = c(rownames_kpSpots_s1, rownames_kpSpots_s2)
  length(rownames_kpSpots)
  
  saveRDS(rownames_kpSpots_s1,
          file=paste0(dir_processedData,"rownames_kpSpots",
                      "_S",sample_s1,"_",sample_s2,"_filterDist005_s1_corrected.rds"))
  saveRDS(rownames_kpSpots_s2,
          file=paste0(dir_processedData,"rownames_kpSpots",
                      "_S",sample_s1,"_",sample_s2,"_filterDist005_s2_corrected.rds"))                                
  
  ### for evaluation
  X_allSpots_s1=read.csv(paste0("X_allSpots_sample",sample_s1,".csv"))
  X_allSpots_s2=read.csv(paste0("X_allSpots_sample",sample_s2,".csv"))
  rownames_allSpots=c(paste0(round(X_allSpots_s1[,2],2),"_",round(X_allSpots_s1[,1],2)),
                      paste0(round(X_allSpots_s2[,2],2),"_",round(X_allSpots_s2[,1],2)))
  saveRDS(rownames_allSpots,
          file=paste0(dir_processedData,"rownames_allSpots",
                      "_S",sample_s1,"_",sample_s2,"_filterDist005_corrected.rds"))

}



########################
########################
for(kpair in 1:9){
  print(kpair)
  sample_s1=list_samplePairs[[kpair]][1]
  sample_s2=list_samplePairs[[kpair]][2]
  
  Xpaste = readRDS(paste0(dir_processedData,"X_allSpots_paste_adj_concatenated_s",sample_s1,"_s",sample_s2,
                      "_filterDist005.rds"))
  
  X_ori_kpSpots_s1 = read.csv(paste0(dir_processedData,"X_kp_s",sample_s1,"_s",sample_s2,
                                     "_filterDist005_s1_corrected.csv"))
  
  X_ori_kpSpots_s2 = read.csv(paste0(dir_processedData,"X_kp_s",sample_s1,"_s",sample_s2,
                                     "_filterDist005_s2_corrected.csv"))
  
  print(nrow(Xpaste))
  print(nrow(X_ori_kpSpots_s1) + nrow(X_ori_kpSpots_s2))
  

}
