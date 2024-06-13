/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.dataset;

import java.awt.image.BufferedImage;
import java.io.IOException;
import static z.dragon.alpha.Alpha.Datas.data;
import static z.dragon.alpha.Alpha.alpha;
import static z.dragon.common.DragonFile.fl;
import z.dragon.data.DataSet;
import z.dragon.data.Transform;
import z.dragon.data.container.ListContainer;

/**
 *
 * @author Gilgamesh
 */
public final class Minist 
{
    private Minist() {}
    
    public static final int num_class = 10;//use one hot coding
    public static final int[] picture_dim = {28, 28, 1};
    
    public static final int test_set_size = 10000;
    public static final String test_image_file = "t10k-images.idx3-ubyte";
    public static final String test_label_file = "t10k-labels.idx1-ubyte";
    
    public static final int train_set_size = 60000;
    public static final String train_image_file = "train-images.idx3-ubyte";
    public static final String train_label_file = "train-labels.idx1-ubyte";
    
    //<editor-fold defaultstate="collapsed" desc="data-load">
    static final Transform<byte[][]> __input_transform(Transform<byte[][]> itf) {
        return (itf == null? data.pixel_to_tensor(picture_dim) : itf);
    }
     
    static final String __directory(String dir) {
        if(dir == null) return alpha.home() + "\\data\\minist\\";
        if(!dir.endsWith("\\")) return dir + "\\";
        return dir;
    } 
    
    static final void load(ListContainer<byte[], Integer> conta, int length, 
            String dir, String imageName, String labelName) throws IOException 
    {
        dir = __directory(dir);
        byte[] imageArr = fl.create(dir + imageName).to_bytes();
        byte[] labelArr = fl.create(dir + labelName).to_bytes();
        
        int imageOffset = 16, labelOffset = 8;
        for(int i=0; i<length; i++)  {
            byte[] pixel = new byte[784]; 
            System.arraycopy(imageArr, imageOffset, pixel, 0, 784);
            imageOffset += 784;
            int label =  labelArr[labelOffset++];
            conta.add(pixel, label);
        }
    }
    
   
    //</editor-fold>
    
    public static DataSet<byte[], Integer> test() { return test(null, null); }
    public static DataSet<byte[], Integer> test(Transform<byte[][]> input_transform) { return test(null, null); }
    public static DataSet<byte[], Integer> test(String dir) { return test(dir, null); }
    public static DataSet<byte[], Integer> test(String dir, Transform<byte[][]> input_transform) {
        ListContainer<byte[], Integer> con = data.list(byte[].class, Integer.class, test_set_size);
        try { load(con, test_set_size, dir, test_image_file, test_label_file); }
        catch(IOException e) { throw new RuntimeException(e); }
        return data.dataset(con)
                .input_transform(__input_transform(input_transform))
                .label_transform(data.onehot(num_class));
    }
    
    public static DataSet<byte[], Integer> train() { return train(null, null); }
    public static DataSet<byte[], Integer> train(Transform<byte[][]> input_transform)  { return train(null, input_transform); }
    public static DataSet<byte[], Integer> train(String dir) { return train(dir, null); }
    public static DataSet<byte[], Integer> train(String dir, Transform<byte[][]> input_transform) {
        ListContainer<byte[], Integer> con = new ListContainer<>(byte[].class, Integer.class, train_set_size);
        try { load(con, train_set_size, dir, train_image_file, train_label_file); }
        catch(IOException e) { throw new RuntimeException(e); }
        if(input_transform == null) input_transform = data.pixel_to_tensor(picture_dim);
        return data.dataset(con)
                .input_transform(__input_transform(input_transform))
                .label_transform(data.onehot(num_class));
    }
}
