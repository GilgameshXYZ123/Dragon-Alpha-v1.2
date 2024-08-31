package Img;

import java.awt.image.BufferedImage;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.common.DragonCV.ConnectedDomain;
import static z.dragon.common.DragonCV.cv;
import z.dragon.common.int32.ArrayList_int32;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;
import z.util.lang.Lang;
import z.util.lang.SimpleTimer;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */



/**
 *
 * @author Gilgamesh
 */
public class connecteArea_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2"); }
    static final Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    public static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    public static void test1() {
        Tensor X = eg.img.read_zip_pixels("C:\\Users\\Gilgamesh\\Desktop\\X0.zip");
        X = eg.img.exp(true, 0.1f, X, 0);
        int IH = X.dim(0), IW = X.dim(1);
        BufferedImage gray = eg.img.gray(X);
//        cv.imshow(gray);
        
        SimpleTimer st = SimpleTimer.clock();
        ConnectedDomain ct = cv.connected_domain()
                .pixel_predicte((p)-> { return p > 30; })
                .group_predicate((v) -> { return v > 300 && v < 1000; })
                .scan(gray);
        st.record();
        System.out.println(st.timeStamp_dif_millis());
        
        int[] group = ct.ConnectedDomain.this.group;
        
        byte[] pix = new byte[IH * IW];
        Map<Integer, ArrayList_int32> map = ct.map;
        System.out.println("map.size = " + map.size());
        map.forEach((k, v)->{ System.out.println(k + " = " + v.size()); });

        int[] v = Lang.exr().next_int_vector(map.size() + 1, 125, 240); v[0] = 0;
        Set<Integer> vs = new HashSet<>();
        for(int i=0; i<pix.length; i++) {
            if (group[i] != 0) pix[i] = (byte) v[group[i]];
            vs.add(group[i]);
        }
        
        System.out.println(vs);
        BufferedImage g2 = cv.gray(pix, IH, IW);
        cv.imshow(g2);
        
        int[][] center = ct.centers();
        int[][] shape  = ct.shapes();
        
        Tensor[] Ys = eg.img.segment2D(X, center, shape);
        BufferedImage[] imgs = eg.img.BGR(Ys);
        
        cv.imshow(imgs);
    }
    
    public static void main(String[] args) {
        test1();
       
    }
    
}
