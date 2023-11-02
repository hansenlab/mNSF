
### load pacakges 
library(tidyverse)
library(ggplot2)
library(ggforce)#
library(cowplot)
library(RColorBrewer)
library(grid)
require(nnet)
library(matrixStats)

########################################################## 
group.colors <- c(Layer1 = "#FC8D62", Layer2 = "#FFD92F", Layer3 ="#A6D854", Layer4 = "#66C2A5", Layer5 = "#00A9FF",
                  Layer6="#8DA0CB",WM="#E78AC3")
myPalette_ <- colorRampPalette(rev(brewer.pal(11, "Spectral")))
myPalette = scale_fill_brewer(palette = "Set2")

########################################################## 
setwd("/dcs04/hansen/data/ywang/ST/DLPFC/PASTE_out_dist005_szMean/")

########################################################## 
kfactor = 2
list_sample = c(1,5,7,12)

########################################################## get genes associated with factor kfactor
## load gene expression matrix and normalize it for sample 1,5,7,12
dir_processedData = "/dcs04/hansen/data/ywang/ST/DLPFC/processed_Data_keepAll/"
list_count_log2_norm=list()

for(ksample in list_sample){
  if(ksample%%2 == 1 ){
    ksamples_pair = c(ksample,ksample+1)
    s_ = 1
  }else{
    ksamples_pair = c(ksample-1, ksample)
    s_ = 2
    
  }
  # load count data for 500 HVGs
  count_sample_tmp=read.csv(paste0("/dcs04/hansen/data/ywang/ST/DLPFC/processed_Data_keepAll/Y_kp_s",ksamples_pair[1],"_s",ksamples_pair[2],"_filterDist005_s",s_,"_corrected.csv"))
  # transpose the count matrix so that each row is one gene, then normalize by library size
  count_sample_tmp_log2_norm = log2(1+t((count_sample_tmp)/colSums(count_sample_tmp))*10^6)
  colnames(count_sample_tmp_log2_norm) = rownames(count_sample_tmp)
  rownames(count_sample_tmp_log2_norm) = colnames(count_sample_tmp)
  list_count_log2_norm[[as.character(ksample)]] = count_sample_tmp_log2_norm
}

genes = rownames(list_count_log2_norm[[1]])

## load factor matrix for sample 1,5,7,12

list_factor_mat=list()

for(ksample in list_sample){
  if(ksample%%2 == 1 ){
    ksamples_pair = c(ksample,ksample+1)
    s_ = 1
  }else{
    ksamples_pair = c(ksample-1, ksample)
    s_ = 2
    
  }
  factor_mat_sampleTmp = read.csv(paste0("/dcs04/hansen/data/ywang/ST/DLPFC/PASTE_out_keepAll_scTransform/factors_nb_szMean_sample_s",ksample,"_L10.csv"),header=T)
  factor_mat_sampleTmp = factor_mat_sampleTmp[,-1]
  list_factor_mat[[as.character(ksample)]] = factor_mat_sampleTmp 
}


#######################################
#######################################
## get genes associated with factor 5
list_sample_pair=list(c(1,2),
                      c(3,4),c(5,6),c(7,8),c(9,10),c(11,12))

list_cor = list()
for(ksample in list_sample){
  list_cor[[as.character(ksample)]]=cor(list_factor_mat[[as.character(ksample)]][,kfactor],
                          t(list_count_log2_norm[[as.character(ksample)]]))
                                       
                            
}

mat_cor = array(unlist(list_cor),dim=c(500,length(list_cor)))
cor_min=rowMins(mat_cor)

topGenes= genes[order(cor_min,decreasing = T)][1:10]
list_mat_exp_norm_topGenes = list()
for(ksample in list_sample){
  rownames(list_count_log2_norm[[as.character(ksample)]])=genes
  list_mat_exp_norm_topGenes[[as.character(ksample)]]=list_count_log2_norm[[as.character(ksample)]][topGenes,]
}


#######################################
#######################################
## prepare layer info for each sample for plotting
list_layer=list()
list_rownames_kpSpots_=list()


for(kpair in 1:length(list_sample_pair)){
    list_layer[[kpair]]=list()
    list_rownames_kpSpots_[[kpair]]=list()
    
    sample_s1=list_sample_pair[[kpair]][1]
    sample_s2=list_sample_pair[[kpair]][2]
    rownames_allSpots = readRDS(paste0(dir_processedData,"rownames_allSpots",
                                       "_S",sample_s1,"_",sample_s2,"_filterDist005_corrected.rds"))
    
    rownames_kpSpots_s1= readRDS(paste0(dir_processedData,"rownames_kpSpots",
                                        "_S",sample_s1,"_",sample_s2,"_filterDist005_s1_corrected.rds"))
    rownames_kpSpots_s2= readRDS(paste0(dir_processedData,"rownames_kpSpots",
                                        "_S",sample_s1,"_",sample_s2,"_filterDist005_s2_corrected.rds"))
    
    rownames_kpSpots = c(rownames_kpSpots_s1, rownames_kpSpots_s2)
    length(rownames_kpSpots)
    # 7112
    
    # load layers
    # save(layer,
         # file=paste0("layer_sample_Sample",j,".RData"))
    load(paste0("//dcs04/hansen/data/ywang/archive/scRNA/Oct5_2021_Lukas_data_more_Genes/out/layer_sample_Sample",sample_s1,".RData"))
    # save(layer_sample_tmp,file=paste0("layer_sample",j,".RData"))
    layer_sample_tmp__s1=layer[]
    
    load(paste0("//dcs04/hansen/data/ywang/archive/scRNA/Oct5_2021_Lukas_data_more_Genes/out/layer_sample_Sample",sample_s2,".RData"))
    # save(layer_sample_tmp,file=paste0("layer_sample",j,".RData"))
    layer_sample_tmp__s2=layer[]
    
    layer_sample_tmp__=c(layer_sample_tmp__s1,layer_sample_tmp__s2)
    names(layer_sample_tmp__)=rownames_allSpots
    
    layer_sample_tmp__s1=layer_sample_tmp__[rownames_kpSpots_s1]
    layer_sample_tmp__s2=layer_sample_tmp__[rownames_kpSpots_s2]
    
    layer_sample_tmp__ = c(layer_sample_tmp__s1, layer_sample_tmp__s2)
    # length(layer_sample_tmp__)

    list_layer[[kpair]][[1]]=layer_sample_tmp__s1
    list_layer[[kpair]][[2]]=layer_sample_tmp__s2
    

    
    list_rownames_kpSpots_[[kpair]][[1]]=rownames_kpSpots_s1
    list_rownames_kpSpots_[[kpair]][[2]]=rownames_kpSpots_s2
    
  }

#######################################
## function for plotting
#######################################
make_plot<-function(rownames_kpSpots,exp_mat,range_perGene_,
                    layer_sample_tmp___,samplename_){
  plots_l = list()
  for (i in 1:ncol(exp_mat)) {
    # dim(a)
    df_tmp=data.frame( imagerow=as.numeric(sub("[_].*","",rownames_kpSpots)),
                       imagecol=as.numeric(sub(".*[_]","",rownames_kpSpots)),
                       fill_tmp=exp_mat[,i],
                       layer=layer_sample_tmp___)
    df_tmp=df_tmp[!is.na(df_tmp$layer),]
    df_tmp$layer = factor(df_tmp$layer, levels = c('WM',"Layer6","Layer5","Layer4","Layer3", "Layer2","Layer1"))
    
    
    plot_tmp =  ggplot(df_tmp,aes(x=imagecol,y=imagerow,fill=fill_tmp)) +
      # geom_spatial(data=images_tibble[i,], aes(grob=grob), x=0.5, y=0.5)+
      geom_point(shape = 21, colour = "black", size = 2, stroke = NA)+
      coord_cartesian(expand=FALSE)+
      # scale_fill_gradientn(colours = myPalette_(100))+
      scale_fill_gradientn(colours = myPalette_(100), limits=range_perGene_[[i]])+
      
      xlab("") +
      ylab("") +
      ggtitle(paste0(samplename_,", ",colnames(exp_mat)[i]))+
      labs(fill = paste0(" "))+
      theme_set(theme_bw(base_size = 10))+
      theme(panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            panel.background = element_blank(),
            axis.line = element_line(colour = "black"),
            axis.text = element_blank(),
            axis.ticks = element_blank())
    
    # if(type_tmp=="l"){
    plots_l[[i]]=plot_tmp
    # }else{
    
    
  }
  plots_l[[i+1]]=ggplot(df_tmp,aes(x=imagecol,y=imagerow,fill=layer)) +
    geom_point(shape = 21, colour = "black", size = 2, stroke = NA)+
    coord_cartesian(expand=FALSE)+
    xlab("") +
    ylab("") +
    ggtitle(paste0("layer "))+
    labs(fill = paste0(" "))+
    theme_set(theme_bw(base_size = 10))+
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.line = element_line(colour = "black"),
          axis.text = element_blank(),
          axis.ticks = element_blank())+
    
    scale_fill_manual(values=group.colors)
  plots_l
}

#######################################
##  get the range for each gene, used for plotting
range_perGene = list()
for(ksample in list_sample){
  # exp_mat_tmp=read.csv(paste0("/dcs04/hansen/data/ywang/ST/DLPFC/PASTE_out_keepAll_scTransform/factors_nb_szMean_sample_s",ksample,"_L10.csv"),header=T)
  exp_mat_tmp=list_mat_exp_norm_topGenes[[as.character(ksample)]]
  exp_mat_tmp_t = t(exp_mat_tmp)
  colnames(exp_mat_tmp_t) = rownames(exp_mat_tmp)
  for(l in 1:ncol(exp_mat_tmp_t)){
    if(ksample==1){ range_perGene[[l]]=range(exp_mat_tmp_t[,l])}else{
      range_perGene[[l]]=range(c(range_perGene[[l]], exp_mat_tmp_t[,l]))
    }
    
  }
}

##  make plots
pdf(paste0("genes_factor",kfactor,"_L10_mNSF.pdf"),height=3,width=18/6*8*3.2)
  for(ksample in list_sample){
    exp_mat_tmp=list_mat_exp_norm_topGenes[[as.character(ksample)]]
    exp_mat_tmp_t = (t(exp_mat_tmp))
    colnames(exp_mat_tmp_t) = rownames(exp_mat_tmp)
    a_ = exp_mat_tmp_t
    if(ksample%%2 == 1 ){
      ksamples_pair = c(ksample,ksample+1)
      s_ = 1
    }else{
      ksamples_pair = c(ksample-1, ksample)
      s_ = 2
      
    }
    kpair = round((ksample+1)/2)
    layers=list_layer[[kpair]][[s_]]
    rownames_kpSpots=list_rownames_kpSpots_[[kpair]][[s_]]
    p1= make_plot(rownames_kpSpots, a_, range_perGene, layers, paste0("sample ",ksample))
    print(plot_grid(plotlist = p1,nrow=1))
  # }
  }
dev.off()









