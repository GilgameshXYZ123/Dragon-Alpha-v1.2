/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package copy;

import copy.Net.ResNet18;
import java.util.ArrayList;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.simple.blas.Conv3D;
import z.util.lang.Lang;
import z.util.lang.SimpleTimer;

/**
 *
 * @author Gilgamesh
 */
public class copy_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    public static void test1()//copy cost total time
    {
        ArrayList arr = new ArrayList<>();
        ResNet18 net = new ResNet18().init(eg);
        SimpleTimer timer = SimpleTimer.clock();
        for(int i=0; i<2000; i++) {
            try {
                ResNet18 net2 = new ResNet18();
                arr.add(net2);
            } catch (Exception ex) {
                Logger.getLogger(copy_test.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        //deep-clone = 470 ms
        //new: 32 ms
        long div = timer.record().timeStamp_dif_millis();
        System.out.println("div = " + div);
    }
    
    public static void test2() {
        Tensor ts = eg.zeros(100, 100, 100);
        SimpleTimer timer = SimpleTimer.clock();
        try {
            for(int i=0; i<30; i++) {
                Tensor x = Lang.clone(ts);
            }
        }
        
        //90
        catch(Exception e) { throw new RuntimeException(e); }
        long div = timer.record().timeStamp_dif_millis();
        System.out.println("div = " + div);
    }
    
    public static void test3() throws Exception {
        Tensor a = eg.zeros(100, 100, 100);
        Tensor b = Lang.clone(a);
        System.out.println(a);
        System.out.println(b);
        System.out.println(a.engine() == b.engine());
    }
    
    public static void test4() throws Exception {
        Conv3D conv1 = alpha.nn.conv3D(false, 32, 32, 3, 2, 1).init(eg);
        Conv3D conv2 = Lang.clone(conv1);
        
        System.out.println(conv1);
        System.out.println(conv2);
        
        System.out.println(conv1.weight());
        System.out.println(conv2.weight());
        System.out.println(conv1.weight() == conv2.weight());
    }
    
    public static void test5() throws Exception {
        ResNet18 net1 = new ResNet18().init(eg);
        ResNet18 net2 = Lang.clone(net1);
        
        Set<Unit> unit1 = net1.units();
        Set<Unit> unit2 = net2.units();
        
        for(Unit u : unit1) {
            System.out.println(unit2.contains(u));
        }
    }
    
    public static void test6() throws Exception {
        Conv3D conv1 = alpha.nn.conv3D(false, 32, 32, 3, 2, 1).init(eg);
        Conv3D conv2 = alpha.nn.conv3D(true, 3, 3, 3, 3, 3).init(eg);
        
        ArrayList<Conv3D> arr1 = new ArrayList<>(2);
        arr1.add(conv1);
        arr1.add(conv2);
        
        ArrayList<Conv3D> arr2 = Lang.clone(arr1);
        Conv3D conv3 = arr2.get(0);
        Conv3D conv4 = arr2.get(1);
        
        System.out.println(conv1 == conv3);
        System.out.println(conv3 == conv4);
    }
    
    public static void main(String[] args) throws Exception
    {
        test1();
    }
}
