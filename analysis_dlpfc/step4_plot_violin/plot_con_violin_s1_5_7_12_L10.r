
### load packages 
library(tidyverse)
library(ggplot2)
library(Matrix)
library(ggforce)#
library(cowplot)
library(RColorBrewer)
library(grid)
require(nnet)

#######

myPalette_ <- colorRampPalette(rev(brewer.pal(11, "Spectral")))
# https://ggplot2-book.org/scale-colour.html
myPalette = scale_fill_brewer(palette = "Set2")

dodge <- position_dodge(width = 3)
group.colors <- c(Layer1 = "#FC8D62", Layer2 = "#FFD92F", Layer3 ="#A6D854", Layer4 = "#66C2A5", Layer5 = "#00A9FF",
                  Layer6="#8DA0CB",WM="#E78AC3")
#######
setwd("/dcs04/hansen/data/ywang/ST/DLPFC/PASTE_out_dist005_szMean/")
dir_processedData = "/dcs04/hansen/data/ywang/ST/DLPFC/processed_Data_keepAll/"
list_sample_pair=list(c(1,2),c(5,6),c(7,8),c(11,12))

#######################################
list_layer=list()

####################################### s1
for(kpair in 1:length(list_sample_pair)){
  list_layer[[kpair]]=list()
  list_rownames_kpSpots_[[kpair]]=list()
  
  sample_s1=list_sample_pair[[kpair]][1]
  sample_s2=list_sample_pair[[kpair]][2]
  # rownames_allSpots = readRDS(paste0(dir_processedData,"rownames_allSpots",
  # "_S",sample_s1,"_",sample_s2,"_filterDist005_corrected.rds"))
  
  # rownames_kpSpots_s1= readRDS(paste0(dir_processedData,"rownames_kpSpots",
  #                                     "_S",sample_s1,"_",sample_s2,"_filterDist005_s1_corrected.rds"))
  # rownames_kpSpots_s2= readRDS(paste0(dir_processedData,"rownames_kpSpots",
  #                                     "_S",sample_s1,"_",sample_s2,"_filterDist005_s2_corrected.rds"))
  # 
  # rownames_kpSpots = c(rownames_kpSpots_s1, rownames_kpSpots_s2)
  # length(rownames_kpSpots)
  # # 7112
  
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
  # names(layer_sample_tmp__)=rownames_allSpots
  
  # layer_sample_tmp__s1=layer_sample_tmp__[rownames_kpSpots_s1]
  # layer_sample_tmp__s2=layer_sample_tmp__[rownames_kpSpots_s2]
  
  # layer_sample_tmp__ = c(layer_sample_tmp__s1, layer_sample_tmp__s2)
  # length(layer_sample_tmp__)
  
  list_layer[[kpair]][[1]]=layer_sample_tmp__s1
  list_layer[[kpair]][[2]]=layer_sample_tmp__s2
  
  
  
  # list_rownames_kpSpots_[[kpair]][[1]]=rownames_kpSpots_s1
  # list_rownames_kpSpots_[[kpair]][[2]]=rownames_kpSpots_s2
  
}

  # load mNSF factors
  a1_=read.csv(paste0("/dcs04/hansen/data/ywang/ST/DLPFC/PASTE_out_keepAll_scTransform/factors_nb_szMean_sample_s",1,"_L10_fullData.csv"),header=T)
  a2_=read.csv(paste0("/dcs04/hansen/data/ywang/ST/DLPFC/PASTE_out_keepAll_scTransform/factors_nb_szMean_sample_s",5,"_L10_fullData.csv"),header=T)
  a3_=read.csv(paste0("/dcs04/hansen/data/ywang/ST/DLPFC/PASTE_out_keepAll_scTransform/factors_nb_szMean_sample_s",7,"_L10_fullData.csv"),header=T)
  a4_=read.csv(paste0("/dcs04/hansen/data/ywang/ST/DLPFC/PASTE_out_keepAll_scTransform/factors_nb_szMean_sample_s",12,"_L10_fullData.csv"),header=T)



  ############ plot
  make_plot_vio<-function(factor_mat,layer_sample_tmp___){
    plots_l = list()
    
    
    for (i in 1:10) {
      
      df_tmp=data.frame(#imagecol=X[,2],
        #imagerow=X[,3],
        layer=layer_sample_tmp___,
        value=factor_mat[,i]
      )
      
      myPalette <- colorRampPalette(rev(brewer.pal(11, "Spectral")))
      
      plot_tmp =  ggplot(df_tmp[!is.na(df_tmp$layer),],aes(x=layer,y=value)) +#,fill=layer,,group=layer
        geom_violin(position = dodge)+
        geom_boxplot(width=.3, outlier.colour=NA, position = dodge)+#+
        # geom_spatial(data=images_tibble[i,], aes(grob=grob), x=0.5, y=0.5)+
        # geom_jitter(shape = 21,size = 0.1, stroke = 0.5,aes(color = value),alpha=0.6 )+#colour = "black", 
        # geom_point(shape = 21, colour = "black", size = .7, stroke = 0.5)+
        geom_jitter(shape = 21, colour = "black", size = .3, stroke = 0.5,alpha=0.7)+
        
        xlab("") +
        ylab("") +
        coord_cartesian(expand=FALSE)+
        
        ggtitle(paste0("factor ",i))+
        # scale_fill_manual(values=group.colors)+#+  geom_jitter()
        # scale_fill_gradientn(colours = myPalette(100))
        scale_fill_gradientn(colours = myPalette(100))
      
      plots_l[[i]]=plot_tmp+coord_flip() 
      
      
    }
    
    plots_l
  }

  
  p1= make_plot_vio( a1_[-1], list_layer[[1]][[1]])
  p2= make_plot_vio( a2_[-1], list_layer[[2]][[1]])
  p3= make_plot_vio( a3_[-1], list_layer[[3]][[1]])
  p4= make_plot_vio( a4_[-1], list_layer[[4]][[2]])
  
  setwd("/dcs04/hansen/data/ywang/ST/DLPFC/PASTE_out_dist005_szMean/")
  
  pdf(paste0( "sample_1_LdaFalse_L10_szMean_violin_fullData.pdf"),height=3,width=10/3*10)
  print(plot_grid(plotlist = p1,nrow=1))
  dev.off()
  
  pdf(paste0( "sample_5_LdaFalse_L10_szMean_violin_fullData.pdf"),height=3,width=10/3*10)
  print(plot_grid(plotlist = p2,nrow=1))
  dev.off()
  
  pdf(paste0( "sample_7_LdaFalse_L10_szMean_violin_fullData.pdf"),height=3,width=10/3*10)
  print(plot_grid(plotlist = p3,nrow=1))
  dev.off()
  
  pdf(paste0( "sample_12_LdaFalse_L10_szMean_violin_fullData.pdf"),height=3,width=10/3*10)
  print(plot_grid(plotlist = p4,nrow=1))
  dev.off()
  
