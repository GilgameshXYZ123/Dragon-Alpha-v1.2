/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Raw;

import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import static z.dragon.alpha.Alpha.alpha;
import static z.dragon.common.DragonCV.cv;
import static z.dragon.common.DragonFile.fl;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;
import z.util.lang.SimpleTimer;

/**
 *
 * @author Gilgamesh
 */
public class raw3
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    static String path = "G:\\pepper-raw\\A.raw";
    
    public static byte[] process_bytes(byte[] X) {
        byte[] Y = new byte[X.length >> 1]; int index = 0;//16 bit -> 8bit
        for(int i=0; i<X.length; i+=2) {
            int b0 = X[i    ] & 0xff;
            int b1 = X[i + 1] & 0xff;
            int b = (b1 << 8) + b0;
            int y = b >> 4;
            Y[index++] = (byte) y;
        }
        return Y;
    }
    
    public static byte[] readBytes(File file) throws IOException {
            byte[] pixel = null; int index = 0;
            FileInputStream in = null;
            BufferedInputStream bfin = null;
            try 
            {
                in = new FileInputStream(file);
                bfin = new BufferedInputStream(in);
                    
                int file_length = (int)file.length();
                if(file_length % 2 != 0) throw new IllegalArgumentException();
                pixel = new byte[file_length >> 1];
                
                int offset = 0, len, buf_size = 4096;
                byte[] buf = new byte[buf_size];
                int read_size = Math.min(buf.length, buf_size);
                while((len = bfin.read(buf, 0, read_size)) != -1 && read_size != 0) 
                {
                    for(int i=0; i<len; i+=2) {
                        int b0 = buf[i    ] & 0xff;
                        int b1 = buf[i + 1] & 0xff;
                        int b = (b1 << 8) + b0;
                        int y = b >> 4;
                        pixel[index++] = (byte) y;
                    }
                    
                    offset += len;
                    read_size = file_length - offset;
                    if(read_size > buf_size) read_size = buf_size;
                }
            }
            catch(IOException e) { pixel = null; throw e; }
            finally {
                if(bfin != null) bfin.close();
                if(in != null) in.close();
            }
            
            return pixel;
    }
    
    //[(IW, IC), IH] -> [IH, (IW, IC)]
    public static byte[] transpose(byte[] X, int N, int M) {
        byte[] Y = new byte[X.length];//X[N, M] -> Y[M, N]
        for(int n=0; n<N; n++)
        for(int m=0; m<M; m++)
            Y[m*N + n] = X[n*M + m]; 
        return Y;
    }
    
    public static void test1() throws IOException  {
        File f = new File(path);
        System.out.println(f.exists());
        System.out.println(f.length());
          
        byte[] bytes = process_bytes(fl.to_bytes(f));
        int height = 640, width = 693, channel = 224;
        
        Tensor X = eg.tensor_int8(bytes, width, channel, height)
                .img().pixel_to_dtype(true)
                .transpose(true, 0, 2)
                .transpose(true, 1, 2)
                .img().dtype_to_pixel(true);
        
        X.img().affine().shear(-0.3f, -0.3f).transform(X, true, height, width);

        Tensor rgb = X.img().extract_3channels(false, 13, 80, 132);
        
        Tensor fX = X.img().pixel_to_dtype(true);
        Tensor gray = eg.row_mean(fX, fX.lastDim()).img().dtype_to_pixel(true);
        
        BufferedImage img1 = cv.gray(gray.pixel(), height, width); cv.imshow(img1, "img1");
        BufferedImage img2 = cv.RGB(rgb.pixel(), height, width); cv.imshow(img2, "img2");
       
        System.gc();
    }
    
    public static void test2() throws IOException  {
        File file = new File(path);
        System.out.println(file.exists());
        System.out.println(file.length());
          
        int IH = 640, IW = 696, IC = 224;
//        byte[] bytes = readBytes(file);//0.13
        
        Tensor X = eg.img.read_raw_bil_dtype12(file, IH, IW, IC).c(); 
        
        X = eg.img.trim2D(true, X, new int[] {150, 300}, new int[] {100, 100});
//        X = eg.img.adjust_brightness(true, X, 0.2f);
        X = eg.img.adjust_constrast(true, X, 0.7f);
//        X = eg.img.adjust_saturation(true, X, -0.8f);
        X = eg.img.affine()
                .rotate(0.5f)
//                .shear(0.2f, 0.2f)
//                .vertical_flip(IH)
//                .horizontal_flip(IW)
                .transform(X, true);
                
        Tensor rgb = X.img().extract_3channels(false, 13, 80, 132);
        
//        Tensor fX = X.img().pixel_to_dtype(true);
//        Tensor gray = eg.row_mean(fX, fX.lastDim()).img().dtype_to_pixel(true);
        
//        BufferedImage img1 = cv.gray(gray.pixel(), IH, IW); cv.imshow(img1, "img1");
        BufferedImage img2 = cv.RGB(rgb); cv.imshow(img2, "img2");
       
        System.gc();
    }
    
    public static void testSpeed(int nIter) throws IOException
    {
        //read file to bytes: time = 0.11s
        //process bytes:      time = 0.06s
        //cretae tensor:      time = 0.0159s
        //tensor transpose:   time = 0.046
        //total: time = 0.237s
        //total: time = 0.15s -> 0.138

        File file = new File(path);
        System.out.println(file.length());
        
        int IH = 640, IW = 693, IC = 224;
        
//        byte[] bytes = process_bytes(fl.to_bytes(file));
//        System.out.println(bytes.length);
        
        SimpleTimer timer = new SimpleTimer().record();
        eg.check(false).sync(false);
        for(int i=0; i<nIter; i++) 
        {
//            byte[] bytes = process_bytes(fl.to_bytes(file));//017
//            Tensor X = eg.tensor_int8(bytes, IW, IC, IH) //[IW, IC, IH]
//                    .img().pixel_to_dtype(true)
//                    .transpose(true, 0, 2)
//                    .transpose(true, 1, 2)
//                    .img().dtype_to_pixel(true);
//            X.delete();

//            byte[] bytes = cv.read_raw_bli_uint16(file);
//            byte[] bytes = cv.read_raw_bli_uint16(path);
//            Tensor X = eg.tensor_int8(bytes, IW*IC, IH) //[IW, IC, IH]
//                    .img().transpose(true, 0, 1)
//                    .view(true, IH, IW, IC); //[IW, IC, IH]
//            X.delete();
            
            Tensor X = eg.img.read_raw_bil_dtype12(path, IH, IW, IC).c();
            X.delete();
        }
        long dif = timer.record().timeStamp_dif_millis();
        float time = 1.0f * dif / 1000 / nIter;
        System.out.println("time = " + time);
    }
     
    public static void main(String[] args)
    {
        try
        {
//            testSpeed(30);
            test2();
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
    }
    
}
