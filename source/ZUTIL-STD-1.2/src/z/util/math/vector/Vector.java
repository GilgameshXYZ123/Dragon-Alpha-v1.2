/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.math.vector;

import java.io.PrintStream;
import java.lang.reflect.Array;
import java.util.Collection;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.function.BiPredicate;
import java.util.function.IntFunction;
import java.util.function.Predicate;
import z.util.ds.linear.ZArrayList;
import z.util.function.SequenceCreator;
import static z.util.lang.Lang.NULL;
import static z.util.lang.Lang.NULL_LN;
import static z.util.lang.Lang.exr;
import z.util.lang.annotation.Passed;
import z.util.lang.exception.IAE;
import z.util.math.Sort;

/**
 *
 * @author dell
 */
public final class Vector
{
    private Vector() {}
    
    //<editor-fold defaultstate="collapsed" desc="String functions">
    public static String double_format = "% 10f";
    public static String double_seperator = ",";
    
    public static String float_format = "% 6f";
    public static String float_seperator = ",";
    
    public static String long_format = "%d";
    public static String long_seperator = ",";
    
    public static String int_format = "%d";
    public static String int_seperator = ",";
    
    public static String short_format = "%d";
    public static String short_seperator = ",";
    
    public static String char_format = "%c";
    public static String char_seperator = ",";
    
    public static String byte_format = "%d";
    public static String byte_seperator = ",";
    
    public static String boolean_format = "%b";
    public static String boolean_seperator = ",";
    
    //<editor-fold defaultstate="collapsed" desc="string functions: append">
    //<editor-fold defaultstate="collapsed" desc="string functions: print<double>">
    public static void append(StringBuilder sb, String msg, double... v) { sb.append(msg); append(sb, v, 0, v.length - 1);} 
    public static void append(StringBuilder sb, String msg, double[] v, int low, int high) { sb.append(msg); append(sb, v, low, high); }
   
    public static void append(StringBuilder sb, double... v) { append(sb, v, 0, v.length - 1); }
    public static void append(StringBuilder sb, double[] v, int low, int high) {
        if(v == null) { sb.append(NULL); return; }
        if(low > high) { int t = low; low = high; high = t; }
        if(low < 0) low = 0;
        if(high >= v.length) high = v.length - 1;
        
        sb.append(String.format(double_format, v[0]));
        for(int i=low+1; i<= high; i++) { 
            sb.append(double_seperator); 
            sb.append(String.format(double_format, v[i])); 
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="string functions: print<float>">
    public static void append(StringBuilder sb, String msg, float... v) { sb.append(msg); append(sb, v, 0, v.length - 1);} 
    public static void append(StringBuilder sb, String msg, float[] v, int low, int high) { sb.append(msg); append(sb, v, low, high); }
   
    public static void append(StringBuilder sb, float... v) { append(sb, v, 0, v.length - 1); }
    public static void append(StringBuilder sb, float[] v, int low, int high) {
        if(v == null) { sb.append(NULL); return; }
        if(low > high) { int t = low; low = high; high = t; }
        if(low < 0) low = 0;
        if(high >= v.length) high = v.length - 1;
        
        sb.append(String.format(float_format, v[0]));
        for(int i=low+1; i<= high; i++) { 
            sb.append(float_seperator); 
            sb.append(String.format(float_format, v[i])); 
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="string functions: print<long>">
    public static void append(StringBuilder sb, String msg, long... v) { sb.append(msg); append(sb, v, 0, v.length - 1);} 
    public static void append(StringBuilder sb, String msg, long[] v, int low, int high) { sb.append(msg); append(sb, v, low, high); }
   
    public static void append(StringBuilder sb, long... v) { append(sb, v, 0, v.length - 1); }
    public static void append(StringBuilder sb, long[] v, int low, int high) {
        if(v == null) { sb.append(NULL); return; }
        if(low > high) { int t = low; low = high; high = t; }
        if(low < 0) low = 0;
        if(high >= v.length) high = v.length - 1;
        
        sb.append(String.format(long_format, v[0]));
        for(int i=low+1; i<= high; i++) { 
            sb.append(long_seperator); 
            sb.append(String.format(long_format, v[i])); 
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="string functions: print<int>">
    public static void append(StringBuilder sb, String msg, int... v) { sb.append(msg); append(sb, v, 0, v.length - 1);} 
    public static void append(StringBuilder sb, String msg, int[] v, int low, int high) { sb.append(msg); append(sb, v, low, high); }
   
    public static void append(StringBuilder sb, int... v) { append(sb, v, 0, v.length - 1); }
    public static void append(StringBuilder sb, int[] v, int low, int high) {
        if(v == null) { sb.append(NULL); return; }
        if(low > high) { int t = low; low = high; high = t; }
        if(low < 0) low = 0;
        if(high >= v.length) high = v.length - 1;
        
        sb.append(String.format(int_format, v[0]));
        for(int i=low+1; i<= high; i++) { 
            sb.append(int_seperator); 
            sb.append(String.format(int_format, v[i])); 
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="string functions: print<short>">
    public static void append(StringBuilder sb, String msg, short... v) { sb.append(msg); append(sb, v, 0, v.length - 1);} 
    public static void append(StringBuilder sb, String msg, short[] v, int low, int high) { sb.append(msg); append(sb, v, low, high); }
   
    public static void append(StringBuilder sb, short... v) { append(sb, v, 0, v.length - 1); }
    public static void append(StringBuilder sb, short[] v, int low, int high) {
        if(v == null) { sb.append(NULL); return; }
        if(low > high) { int t = low; low = high; high = t; }
        if(low < 0) low = 0;
        if(high >= v.length) high = v.length - 1;
        
        sb.append(String.format(short_format, v[0]));
        for(int i=low+1; i<= high; i++) { 
            sb.append(short_seperator); 
            sb.append(String.format(short_format, v[i])); 
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="string functions: print<char>">
    public static void append(StringBuilder sb, String msg, char... v) { sb.append(msg); append(sb, v, 0, v.length - 1);} 
    public static void append(StringBuilder sb, String msg, char[] v, int low, int high) { sb.append(msg); append(sb, v, low, high); }
   
    public static void append(StringBuilder sb, char... v) { append(sb, v, 0, v.length - 1); }
    public static void append(StringBuilder sb, char[] v, int low, int high) {
        if(v == null) { sb.append(NULL); return; }
        if(low > high) { int t = low; low = high; high = t; }
        if(low < 0) low = 0;
        if(high >= v.length) high = v.length - 1;
        
        sb.append(String.format(char_format, v[0]));
        for(int i=low+1; i<= high; i++) { 
            sb.append(char_seperator); 
            sb.append(String.format(char_format, v[i])); 
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="string functions: print<byte>">
    public static void append(StringBuilder sb, String msg, byte[]... v) { sb.append(msg); append(sb, v, 0, v.length - 1);} 
    public static void append(StringBuilder sb, String msg, byte[] v, int low, int high) { sb.append(msg); append(sb, v, low, high); }
   
    public static void append(StringBuilder sb, byte... v) { append(sb, v, 0, v.length - 1); }
    public static void append(StringBuilder sb, byte[] v, int low, int high) {
        if(v == null) { sb.append(NULL); return; }
        if(low > high) { int t = low; low = high; high = t; }
        if(low < 0) low = 0;
        if(high >= v.length) high = v.length - 1;
        
        sb.append(String.format(byte_format, v[0]));
        for(int i=low+1; i<= high; i++) { 
            sb.append(byte_seperator); 
            sb.append(String.format(byte_format, v[i])); 
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="string functions: print<boolean>">
    public static void append(StringBuilder sb, String msg, boolean... v) { sb.append(msg); append(sb, v, 0, v.length - 1); }
    public static void append(StringBuilder sb, String msg, boolean[] v, int low, int high) { sb.append(msg); append(sb, v, low, high); }
   
    public static void append(StringBuilder sb, boolean... v) { append(sb, v, 0, v.length - 1); }
    public static void append(StringBuilder sb, boolean[] v, int low, int high) {
        if(v == null) { sb.append(NULL); return; }
        if(low > high) { int t = low; low = high; high = t; }
        if(low < 0) low = 0;
        if(high >= v.length) high = v.length - 1;
        
        sb.append(String.format(boolean_format, v[0]));
        for(int i=low+1; i<= high; i++) { 
            sb.append(boolean_seperator); 
            sb.append(String.format(boolean_format, v[i])); 
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="string functions: print<Object>">
    public static void append(StringBuilder sb, String msg, Object[] v) { sb.append(msg); append(sb, v, 0, v.length - 1);} 
    public static void append(StringBuilder sb, String msg, Object[] v, int low, int high) { sb.append(msg); append(sb, v, low, high); }
   
    public static void append(StringBuilder sb, Object[] v) { append(sb, v, 0, v.length - 1); }
    public static void append(StringBuilder sb, Object[] v, int low, int high) {
        if(v == null) { sb.append(NULL); return; }
        if(low > high) { int t = low; low = high; high = t; }
        if(low < 0) low = 0;
        if(high >= v.length) high = v.length - 1;
        
        sb.append(String.format(object_format, v[0]));
        for(int i=low+1; i<= high; i++) { 
            sb.append(object_seperator); 
            sb.append(String.format(object_format, v[i])); 
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="string functions: print<extensive>">
    public static void append(StringBuilder sb, Collection v) {
        if(v==null) {sb.append(NULL);return;}
        for(Object val:v) sb.append(val).append(',');
        if(sb.charAt(sb.length()-1)==',') sb.setLength(sb.length()-1);
    }

    public static void appendLn(StringBuilder sb, Collection v) {
        if(v == null) {sb.append(NULL_LN);return;}
        for(Object o:v) sb.append(o).append('\n');
    }
    public static void appendLn(StringBuilder sb, Collection v, String prefix) {
        if(v == null) {sb.append(NULL_LN); return; }
        for(Object o:v) sb.append(prefix).append(o).append('\n');
    }
    
    public static void appendLn(StringBuilder sb, Map map) {
        if(map == null) { sb.append(NULL_LN); return; }
        map.forEach((Object key, Object value)->{
            sb.append('\n').append(key).append(" = ").append(value);
        });
    }
    public static void appendLn(StringBuilder sb, Map map, String prefix) {
        if(map == null) {sb.append(NULL_LN);return;}
        map.forEach((Object key, Object value) -> { 
            sb.append(prefix).append(key).append(" = ").append(value);
        });
    }
    
    public static void appendLimitCapacity(StringBuilder sb, float[] value) {
        int capacity = sb.capacity() - 10, index = 0, lineFeed = 0;
        while(index<value.length-1 && sb.length()<capacity) {
            sb.append(value[index]).append(',');
            index++; lineFeed++;
            if((lineFeed&7) ==0) sb.append("\n\t");
        }
        if(sb.length()<capacity) sb.append(value[value.length - 1]);
        if(index<value.length) sb.append(".......");
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="string-function: toString">
    public static String toString(double... v) { return toString(v, 0, v.length - 1); }
    public static String toString(double[] v, int low, int high) {
        if(v == null) return NULL;
        if(low < high) { int t = low; low = high; high = t; }
        StringBuilder sb = new StringBuilder((high - low + 1) << 4);
        Vector.append(sb, v, low, high);
        return sb.toString();
    }
    
    public static String toString(float... v) { return toString(v, 0, v.length - 1); }
    public static String toString(float[] v, int low, int high) {
        if(v == null) return NULL;
        if(low < high) { int t = low; low = high; high = t; }
        StringBuilder sb = new StringBuilder((high - low + 1) << 3);
        Vector.append(sb, v, low, high);
        return sb.toString();
    }
    
    public static String toString(long... v) { return toString(v, 0, v.length - 1); }
    public static String toString(long[] v, int low, int high) {
        if(v == null) return NULL;
        if(low < high) { int t = low; low = high; high = t; }
        StringBuilder sb = new StringBuilder((high - low + 1) << 5);
        Vector.append(sb, v, low, high);
        return sb.toString();
    }
    
    public static String toString(int... v) { return toString(v, 0, v.length - 1); }
    public static String toString(int[] v, int low, int high) {
        if(v == null) return NULL;
        if(low > high) { int t = low; low = high; high = t; }
        StringBuilder sb = new StringBuilder((high - low + 1) << 4);
        Vector.append(sb, v, low, high);
        return sb.toString();
    }
    
    public static String toString(short... v) { return toString(v, 0, v.length - 1); }
    public static String toString(short[] v, int low, int high) {
        if(v == null) return NULL;
        if(low > high) { int t = low; low = high; high = t; }
        StringBuilder sb = new StringBuilder((high - low + 1) << 3);
        Vector.append(sb, v, low, high);
        return sb.toString();
    }
    
    public static String toString(char... v) { return toString(v, 0, v.length - 1); }
    public static String toString(char[] v, int low, int high) {
        if(v == null) return NULL;
        if(low > high) { int t = low; low = high; high = t; }
        StringBuilder sb = new StringBuilder((high - low + 1));
        Vector.append(sb, v, low, high);
        return sb.toString();
    }
    
    public static String toString(byte... v) { return toString(v, 0, v.length - 1); }
    public static String toString(byte[] v, int low, int high) {
        if(v == null) return NULL;
        if(low > high) { int t = low; low = high; high = t; }
        StringBuilder sb = new StringBuilder((high - low + 1) << 3);
        Vector.append(sb, v, low, high);
        return sb.toString();
    }
    
    public static String toString(boolean... v) { return toString(v, 0, v.length - 1); }
    public static String toString(boolean[] v, int low, int high) {
        if(v == null) return NULL;
        if(low > high) { int t = low; low = high; high = t; }
        StringBuilder sb = new StringBuilder((high - low + 1) * 6);
        Vector.append(sb, v, low, high);
        return sb.toString();
    }
  
    public static String toString(Object[] v) {
        StringBuilder sb = new StringBuilder(v.length << 5);
        Vector.append(sb, v);
        return sb.toString();
    }
    
    public static String toStringln(Map m) {
        StringBuilder sb = new StringBuilder(m.size() << 5);
        m.forEach((k,v)->{sb.append('\n').append(k).append(" = ").append(v);});
        return sb.toString();
    }
    
    public static String toStringln(Collection v) {
        StringBuilder sb = new StringBuilder(v.size() << 5);
        for(Object o:v) sb.append(o).append('\n');
        return sb.toString();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="string functions: print">
    private static final PrintStream def_out = System.out;
    public static synchronized void default_printStream(PrintStream out){}
    public static PrintStream default_printStream() { return def_out; }
    
    //<editor-fold defaultstate="collapsed" desc="string functions: print<double>">
    public static void println(String msg, double... v) { def_out.print(msg); println(def_out, v);} 
    public static void println(String msg, double[] v, int low, int high) { def_out.print(msg); println(def_out, v, low, high);  }
   
    public static void println(double... v) { println(def_out, v, 0, v.length - 1);  }
    public static void println(double[] v, int low, int high) { println(def_out, v, low, high); }
    
    public static void println(PrintStream out, double... v) { println(out, v, 0, v.length - 1);  }
    public static void println(PrintStream out, double[] v, int low, int high) {
        if(v == null) { out.println(NULL); return; }
        if(low > high) { int t = low; low = high; high = t; }
        if(low < 0) low = 0;
        if(high >= v.length) high = v.length - 1;
        
        out.format(double_format, v[0]);
        for(int i=low+1; i<= high; i++) { 
            out.print(double_seperator); 
            out.format(double_format, v[i]); 
        }
        out.println();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="string functions: print<float>">
    public static void println(String msg, float... v) { def_out.print(msg); println(def_out, v);} 
    public static void println(String msg, float[] v, int low, int high) { def_out.print(msg); println(def_out, v, low, high);  }
   
    public static void println(float... v) { println(def_out, v, 0, v.length - 1);  }
    public static void println(float[] v, int low, int high) { println(def_out, v, low, high); }
    
    public static void println(PrintStream out, float... v) { println(out, v, 0, v.length - 1);  }
    public static void println(PrintStream out, float[] v, int low, int high) {
        if(v == null) { out.println(NULL); return; }
        if(low > high) { int t = low; low = high; high = t; }
        if(low < 0) low = 0;
        if(high >= v.length) high = v.length - 1;
        
        out.format(float_format, v[0]);
        for(int i=low+1; i<= high; i++) { 
            out.print(float_seperator); 
            out.format(float_format, v[i]); 
        }
        out.println();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="string functions: print<long>">
    public static void println(String msg, long... v) { def_out.print(msg); println(def_out, v);} 
    public static void println(String msg, long[] v, int low, int high) { def_out.print(msg); println(def_out, v, low, high);  }
   
    public static void println(long... v) { println(def_out, v, 0, v.length - 1);  }
    public static void println(long[] v, int low, int high) { println(def_out, v, low, high); }
    
    public static void println(PrintStream out, long... v) { println(out, v, 0, v.length - 1);  }
    public static void println(PrintStream out, long[] v, int low, int high) {
        if(v == null) { out.println(NULL); return; }
        if(low > high) { int t = low; low = high; high = t; }
        if(low < 0) low = 0;
        if(high >= v.length) high = v.length - 1;
        
        out.format(long_format, v[0]);
        for(int i=low+1; i<= high; i++) { 
            out.print(long_seperator); 
            out.format(long_format, v[i]); 
        }
        out.println();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="string functions: print<int>">
    public static void println(String msg, int... v) { def_out.print(msg); println(def_out, v);} 
    public static void println(String msg, int[] v, int low, int high) { def_out.print(msg); println(def_out, v, low, high);  }
   
    public static void println(int... v) { println(def_out, v, 0, v.length - 1);  }
    public static void println(int[] v, int low, int high) { println(def_out, v, low, high); }
    
    public static void println(PrintStream out, int... v) { println(out, v, 0, v.length - 1);  }
    public static void println(PrintStream out, int[] v, int low, int high) {
        if(v == null) { out.println(NULL); return; }
        if(low > high) { int t = low; low = high; high = t; }
        if(low < 0) low = 0;
        if(high >= v.length) high = v.length - 1;
        
        out.format(int_format, v[0]);
        for(int i=low+1; i<= high; i++) { 
            out.print(int_seperator); 
            out.format(int_format, v[i]); 
        }
        out.println();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="string functions: print<short>">
    public static void println(String msg, short... v) { def_out.print(msg); println(def_out, v);} 
    public static void println(String msg, short[] v, int low, int high) { def_out.print(msg); println(def_out, v, low, high);  }
   
    public static void println(short... v) { println(def_out, v, 0, v.length - 1);  }
    public static void println(short[] v, int low, int high) { println(def_out, v, low, high); }
    
    public static void println(PrintStream out, short... v) { println(out, v, 0, v.length - 1);  }
    public static void println(PrintStream out, short[] v, int low, int high) {
        if(v == null) { out.println(NULL); return; }
        if(low > high) { int t = low; low = high; high = t; }
        if(low < 0) low = 0;
        if(high >= v.length) high = v.length - 1;
        
        out.format(short_format, v[0]);
        for(int i=low+1; i<= high; i++) { 
            out.print(short_seperator); 
            out.format(short_format, v[i]); 
        }
        out.println();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="string functions: print<char>">
    public static void println(String msg, char... v) { def_out.print(msg); println(def_out, v);} 
    public static void println(String msg, char[] v, int low, int high) { def_out.print(msg); println(def_out, v, low, high);  }
   
    public static void println(char... v) { println(def_out, v, 0, v.length - 1);  }
    public static void println(char[] v, int low, int high) { println(def_out, v, low, high); }
    
    public static void println(PrintStream out, char... v) { println(out, v, 0, v.length - 1);  }
    public static void println(PrintStream out, char[] v, int low, int high) {
        if(v == null) { out.println(NULL); return; }
        if(low > high) { int t = low; low = high; high = t; }
        if(low < 0) low = 0;
        if(high >= v.length) high = v.length - 1;
        
        out.format(char_format, v[0]);
        for(int i=low+1; i<= high; i++) { 
            out.print(char_seperator); 
            out.format(char_format, v[i]); 
        }
        out.println();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="string functions: print<byte>">
    public static void println(String msg, byte[] v) { def_out.print(msg); println(def_out, v);} 
    public static void println(String msg, byte[] v, int low, int high) { def_out.print(msg); println(def_out, v, low, high);  }
   
    public static void println(byte[] v) { println(def_out, v, 0, v.length - 1);  }
    public static void println(byte[] v, int low, int high) { println(def_out, v, low, high); }
    
    public static void println(PrintStream out, byte[] v) { println(out, v, 0, v.length - 1);  }
    public static void println(PrintStream out, byte[] v, int low, int high) {
        if(v == null) { out.println(NULL); return; }
        if(low > high) { int t = low; low = high; high = t; }
        if(low < 0) low = 0;
        if(high >= v.length) high = v.length - 1;
        
        out.format(byte_format, v[0]);
        for(int i=low+1; i<= high; i++) { 
            out.print(byte_seperator); 
            out.format(byte_format, v[i]); 
        }
        out.println();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="string functions: print<boolean>">
    public static void println(String msg, boolean[] v) { def_out.print(msg); println(def_out, v);} 
    public static void println(String msg, boolean[] v, int low, int high) { def_out.print(msg); println(def_out, v, low, high);  }
   
    public static void println(boolean[] v) { println(def_out, v, 0, v.length - 1);  }
    public static void println(boolean[] v, int low, int high) { println(def_out, v, low, high); }
    
    public static void println(PrintStream out, boolean[] v) { println(out, v, 0, v.length - 1);  }
    public static void println(PrintStream out, boolean[] v, int low, int high) {
        if(v == null) { out.println(NULL); return; }
        if(low > high) { int t = low; low = high; high = t; }
        if(low < 0) low = 0;
        if(high >= v.length) high = v.length - 1;
        
        out.format(boolean_format, v[0]);
        for(int i=low+1; i<= high; i++) { 
            out.print(boolean_seperator); 
            out.format(boolean_format, v[i]); 
        }
        out.println();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="string functionsn: print<Object>">
    public static String object_format = "%s";
    public static String object_seperator = ",";
    
    public static void println(String msg, Object[] v) { def_out.print(msg); println(def_out, v);} 
    public static void println(String msg, Object[] v, int low, int high) { def_out.print(msg); println(def_out, v, low, high);  }
   
    public static void println(Object[] v) { println(def_out, v, 0, v.length - 1);  }
    public static void println(Object[] v, int low, int high) { println(def_out, v, low, high); }
    
    public static void println(PrintStream out, Object[] v) { println(out, v, 0, v.length - 1);  }
    public static void println(PrintStream out, Object[] v, int low, int high) {
        if(v == null) { out.println(NULL); return; }
        if(low > high) { int t = low; low = high; high = t; }
        if(low < 0) low = 0;
        if(high >= v.length) high = v.length - 1;
        
        out.format(object_format, v[0]);
        for(int i=low+1; i<= high; i++) { 
            out.print(object_seperator); 
            out.format(object_format, v[i]); 
        }
        out.println();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="string functionsn: print<extensive>">
    public static void println(Collection v) { println(def_out, v); }
    public static void println(PrintStream out, Collection v) {
        if(v == null) { out.println(NULL); return; }
        for(Object o : v) out.println(o);
    }

    public static void println(Map m) { println(def_out, m); }
    public static void println(PrintStream out, Map m) {
        if(m == null) { out.println(NULL); return; }
        m.forEach((k,v) -> { out.println(k + " = " + v); });
    }
    
     public static void println(Enumeration en) {}
    public static void println(PrintStream out, Enumeration en) {
        if(en == null) { out.println(NULL); return; }
        while(en.hasMoreElements()) out.println(en.nextElement());
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Vector: convert functions">
    //<editor-fold defaultstate="collapsed" desc="convert-functions: string -> vector">
    public static float[] to_float_vector(String str) { return Vector.to_float_vector(str.split(",")); }
    public static float[] to_float_vector(String[] tokens) {
        float[] r = new float[tokens.length];
        for (int i = 0; i < tokens.length; i++) r[i] = Float.valueOf(tokens[i]);
        return r;
    }
    
    public static double[] to_double_vector(String str) { return to_double_vector(str.split(",")); }
    public static double[] to_double_vector(String[] tokens) {
        double[] r = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) r[i] = Double.valueOf(tokens[i]);
        return r;
    }
    
    public static long[] to_long_vector(String str) { return to_long_vector(str.split(",")); }
    public static long[] to_long_vector(String[] tokens) {
        long[] r = new long[tokens.length];
        for (int i = 0; i < tokens.length; i++) r[i] = Long.valueOf(tokens[i]);
        return r;
    }
    
    public static int[] to_int_vector(String str) { return to_int_vector(str.split(",")); }
    public static int[] to_int_vector(String[] tokens) {
        int[] r = new int[tokens.length];
        for (int i = 0; i < tokens.length; i++) r[i] = Integer.valueOf(tokens[i]);
        return r;
    }
    
    public static short[] to_short_vector(String str) { return to_short_vector(str.split(",")); }
    public static short[] to_short_vector(String[] tokens) {
        short[] r = new short[tokens.length];
        for (int i = 0; i < tokens.length; i++) r[i] = Short.valueOf(tokens[i]);
        return r;
    }
    
    public static byte[] to_byte_vector(String str) { return to_byte_vector(str.split(",")); }
    public static byte[] to_byte_vector(String... tokens)  {
        byte[] r = new byte[tokens.length];
        for (int i = 0; i < tokens.length; i++) r[i] = Byte.valueOf(tokens[i]);
        return r;
    }
    
    public static boolean[] to_boolean_vector(String str) { return to_boolean_vector(str.split(",")); }
    public static boolean[] to_boolean_vector(String[] tokens) {
        boolean[] r = new boolean[tokens.length];
        for (int i = 0; i < tokens.length; i++) r[i] = Boolean.valueOf(tokens[i]);
        return r;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="convert-functions: string[] -> vector">
    public static float[] to_float_vector(List<String> lines, int length)  {
        float[] value = new float[length]; int index = 0;
        for(String line : lines) 
            for(String token : line.split(","))
                value[index++] = Float.valueOf(token);
        return value;
    }
    
    public static int[] to_int_vector(List<String> lines, int length)  {
        int[] value = new int[length]; int index = 0;
        for(String line : lines) 
            for(String token : line.split(","))
                value[index++] = Integer.valueOf(token);
        return value;
    }
    //</editor-fold>
    
    public static float[] to_float_vector(byte[] arr) {
        float[] farr = new float[arr.length];
        for(int i=0; i<farr.length; i++) farr[i] = arr[i];
        return farr;
    }
    public static float[] to_float_vector(int[] arr) {
        float[] farr = new float[arr.length];
        for(int i=0; i<farr.length; i++) farr[i] = arr[i];
        return farr;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Array-Convert-Function:numbers">
    public static Integer[] to_Integer_vector(String str) { return to_Integer_vector(str.split(",")); }
    public static Integer[] to_Integer_vector(String[] tokens) {
        Integer[] r = new Integer[tokens.length];
        for (int i = 0; i < tokens.length; i++) r[i] = Integer.valueOf(tokens[i]);
        return r;
    }
    
    public static Short[] to_Short_vector(String str) { return to_Short_vector(str.split(",")); }
    public static Short[] to_Short_vector(String[] tokens) {
        Short[] r = new Short[tokens.length];
        for (int i = 0; i < tokens.length; i++)  r[i] = Short.valueOf(tokens[i]);
        return r;
    }
    
    public static Byte[] to_Byte_vector(String str) { return to_Byte_vector(str.split(",")); }
    public static Byte[] to_Byte_vector(String[] tokens) {
        Byte[] r = new Byte[tokens.length];
        for (int i = 0; i < tokens.length; i++)  r[i] = Byte.valueOf(tokens[i]);
        return r;
    }
    
    public static Boolean[] to_Boolean_vector(String str) { return to_Boolean_Vector(str.split(",")); }
    public static Boolean[] to_Boolean_Vector(String[] tokens) {
        Boolean[] r = new Boolean[tokens.length];
        for (int i = 0; i < tokens.length; i++) r[i] = Boolean.valueOf(tokens[i]);
        return r;
    }
    
    public static Float[] to_Float_vector(String str) { return to_Float_vector(str.split(",")); }
    public static Float[] to_Float_vector(String[] tokens) {
        Float[] r = new Float[tokens.length];
        for (int i = 0; i < tokens.length; i++)  r[i] = Float.valueOf(tokens[i]);
        return r;
    }
    
    public static Double[] to_Double_vector(String str) { return Vector.to_Double_vector(str.split(",")); }
    public static Double[] to_Double_vector(String[] tokens) {
        Double[] r = new Double[tokens.length];
        for (int i = 0; i < tokens.length; i++) r[i] = Double.valueOf(tokens[i]);
        return r;
    }
    
    public static Long[] to_Long_vector(String str) { return to_Long_vector(str.split(",")); }
    public static Long[] to_Long_vector(String[] tokens) {
        Long[] r = new Long[tokens.length];
        for (int i = 0; i < tokens.length; i++) r[i] = Long.valueOf(tokens[i]);
        return r;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="vector: math function">
    public static <T> T first_nonNull(T[] arr) { return first_nonNull(arr, 0); }
    public static <T> T first_nonNull(T[] arr, int index) {
        if(arr == null) return null;
        for(int i = index; i<arr.length; i++) 
            if(arr[i] != null) return arr[i];
        return null;
    }
    
    public static <T> void putFirstNoNullToHead(T[] val) {
        T fn=Vector.first_nonNull(val);
        T t=fn;fn=val[0];val[0]=t;
    }
    public static <T> void putFirstNoNullToHead(T[] val, int index)
    {
        T fn=Vector.first_nonNull(val, index);
        T t=fn;fn=val[index];val[index]=t;
    }
    
    //<editor-fold defaultstate="collapsed" desc="math: minValue">
    public static float minValue(float... arr) { return minValue(arr, 0, arr.length - 1); }
    public static float minValue(float[] arr, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        float min = arr[low];
        for(int i=low+1; i<=high; i++) if(min > arr[i]) min = arr[i];
        return min;
    }
    
    public static double minValue(double... arr) { return Vector.minValue(arr, 0, arr.length - 1); }
    public static double minValue(double[] arr, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        double min = arr[low];
        for(int i=low+1; i<=high; i++) if(min > arr[i]) min = arr[i];
        return min;
    }
    
    public static int minValue(byte[] arr) { return Vector.minValue(arr, 0, arr.length-1); }
    public static int minValue(byte[] arr, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        byte min = arr[low];
        for(int i=1+low; i<=high; i++) if(min > arr[i]) min = arr[i];
        return min;
    }
    
    public static char minValue(char... arr) { return Vector.minValue(arr, 0, arr.length - 1); }
    public static char minValue(char[] arr, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        char min = arr[low];
        for(int i=1+low; i<=high; i++) if(min > arr[i]) min = arr[i];
        return min;
    }
    
    public static int minValue(int... arr) { return Vector.minValue(arr, 0, arr.length - 1); }
    public static int minValue(int[] arr, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        int min = arr[low];
        for(int i=1+low; i<=high; i++) if(min > arr[i]) min = arr[i];
        return min;
    }
    
    public static long minValue(long... arr) { return Vector.minValue(arr, 0, arr.length - 1); }
    public static long minValue(long[] arr, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        long min = arr[low];
        for(int i=1+low; i<=high; i++) if(min > arr[i]) min = arr[i];
        return min;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="math: maxValue">
    public static float maxValue(float... arr) { return maxValue(arr, 0, arr.length - 1); }
    public static float maxValue(float[] arr, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        float max = arr[low];
        for(int i=low+1; i<=high; i++) if(max < arr[i]) max = arr[i];
        return max;
    }
    
    public static double maxValue(double... arr) { return maxValue(arr, 0, arr.length - 1); }
    public static double maxValue(double[] arr,int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        double max = arr[low];
        for(int i=low+1; i<=high; i++) if(max < arr[i]) max = arr[i];
        return max;
    }
    
    public static byte maxValue(byte[] arr) { return maxValue(arr, 0, arr.length - 1); }
    public static byte maxValue(byte[] arr, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        byte max = arr[0];
        for(int i=low+1; i<=high; i++) if(max < arr[i]) max = arr[i];
        return max;
    }
   
    public static char maxValue(char[] arr) { return maxValue(arr, 0, arr.length - 1); }
    public static char maxValue(char[] arr, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        char max = arr[0];
        for(int i=low+1; i<=high; i++) if(max < arr[i]) max = arr[i];
        return max;
    }
    
    public static int maxValue(int[] arr) { return maxValue(arr, 0, arr.length - 1); }
    public static int maxValue(int[] arr, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        int max = arr[0];
        for(int i=low+1; i<=high; i++) if(max < arr[i]) max = arr[i];
        return max;
    }
    
    public static long maxValue(long[] arr) { return maxValue(arr, 0, arr.length - 1); }
    public static long maxValue(long[] arr, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        long max = arr[0];
        for(int i=low+1; i<=high; i++) if(max < arr[i]) max = arr[i];
        return max;
    }
    //</editor-fold>
    
    public static <T extends Comparable> T max(T[] val) {
        Vector.putFirstNoNullToHead(val);
        T max=val[0]; 
        if(val[0]==null) return null;
        for(int i=1;i<val.length;i++)
            if(max.compareTo(val[i])<0) max=val[i];
        return max;
    }
     
    public static int minValueIndex(int[] val) { return minValueIndex(val, 0, val.length-1); }
    public static int minValueIndex(int[] val, int low, int high) {
        int min = val[low], index=0;
        for(int i=low+1; i<=high; i++) 
            if(min > val[i]) { min = val[i]; index = i; }
        return index;
    }
   
    public static int minValueIndex(float[] val, int low, int high) {
        float min = val[low]; int index=0;
        for(int i=low+1; i<=high; i++) 
            if(min > val[i]) { min = val[i]; index = i; }
        return index;
    }
    public static int minValueIndex(float[] val) {return Vector.minValueIndex(val, 0, val.length-1);}
    
    @Passed
    public static <T extends Comparable> T minValue(T[] val)
    {
        Vector.putFirstNoNullToHead(val);
        T min=val[0];
        if(val[0]==null) return null;
        for(int i=1;i<val.length;i++)
            if(min.compareTo(val[i])>0) min=val[i];
        return min;
    }
    
    @Passed
    public static int maxValueIndex(int[] val, int low, int high) {
        int max = val[low], index = low;
        for(int i=low+1; i<=high; i++) 
            if(max < val[i]) { max = val[i]; index = i; }
        return index;
    }
    public static int maxValueIndex(int[] val) {
        return Vector.maxValueIndex(val, 0, val.length-1);
    }
    
    @Passed
    public static int maxValueIndex(float[] val, int low, int high) {
        float max = val[low]; int index = low;
        for(int i = low + 1; i<=high; i++) 
            if(max < val[i]) { max = val[i]; index = i; }
        return index;
    }
    public static int maxValueIndex(float[] val) {
        return Vector.maxValueIndex(val, 0, val.length-1);
    }
    
    
    //<editor-fold defaultstate="collapsed" desc="class MaxMin">
    public static final class MaxMin<T> {
        T max, min;
        
        MaxMin() {}
        MaxMin(T max, T min) {
            this.max = max;
            this.min = min;
        }
        
        public T max() { return max; }
        public T min() { return min; }
        @Override public String toString() { return "{ max = " + max + ", min = " + min + " }"; }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Inner-Code:MaxMin">
    @Passed
    static void maxMinABS(int[] val, int low, int high, MaxMin<Integer> mm) {
        int max,min;
        int abs1= (val[low]>=0? val[low]: -val[low]);
        int abs2= (val[low+1]>=0? val[low+1]: -val[low+1]);
        if(abs1>abs2)  {max=abs1;min=abs2;}
        else {max=abs1;min=abs2;}
        for(int i=2+low;i<high;i+=2)
        {
            abs1=(val[i]>=0? val[i]: -val[i]);
            abs2=(val[i+1]>=0? val[i+1]: -val[i+1]);
            if(abs1>abs2)
            {
                if(max<abs1) max=abs1;
                if(min>abs2) min=abs2;
            }
            else
            {
                if(max<abs2) max=abs2;
                if(min>abs1) min=abs1;
            }
        }
        if(((high-low+1)&1)==1)//val.length is an odd number
        {
            int ev=(val[high]>=0? val[high]:-val[high]);
            if(max<ev) max=ev;
            else if(min>ev) min=ev;
        }
        mm.max=max;
        mm.min=min;
    }
    
    static void maxMinABSIndex(int[] val, int low, int high, MaxMin<Integer> mm)
    {
        int max,min, maxIndex, minIndex;
        int abs1= (val[low]>=0? val[low]: -val[low]);
        int abs2= (val[low+1]>=0? val[low+1]: -val[low+1]);
        if(abs1>abs2)  {max=abs1; maxIndex=0; min=abs2; minIndex=1;}
        else {max=abs2; maxIndex=1; min=abs1; minIndex=0;}
        for(int i=2+low;i<high;i+=2)
        {
            abs1=(val[i]>=0? val[i]: -val[i]);
            abs2=(val[i+1]>=0? val[i+1]: -val[i+1]);
            if(abs1>abs2)
            {
                if(max<abs1) {max=abs1; maxIndex=i;}
                if(min>abs2) {min=abs2; minIndex=i+1;}
            }
            else
            {
                if(max<abs2) {max=abs2; maxIndex=i+1;}
                if(min>abs1) {min=abs1; minIndex=i;}
            }
        }
        if(((high-low+1)&1)==1)//val.length is an odd number
        {
            int ev=(val[high]>=0? val[high]:-val[high]);
            if(max<ev) maxIndex=high;
            else if(min>ev) minIndex=high;
        }
        mm.max=maxIndex;
        mm.min=minIndex;
    }

    @Passed
    static void maxMinABS(double[] val, int low, int high, MaxMin<Double> mm)
    {
        double max,min;
        double abs1= (val[low]>=0? val[low]: -val[low]);
        double abs2= (val[low+1]>=0? val[low+1]: -val[low+1]);
        if(abs1>abs2)  {max=abs1;min=abs2;}
        else {max=abs2;min=abs1;}
        for(int i=2+low;i<high;i+=2)
        {
            abs1=(val[i]>=0? val[i]: -val[i]);
            abs2=(val[i+1]>=0? val[i+1]: -val[i+1]);
            if(abs1>abs2)
            {
                if(max<abs1) max=abs1;
                if(min>abs2) min=abs2;
            }
            else
            {
                if(max<abs2) max=abs2;
                if(min>abs1) min=abs1;
            }
        }
        if(((high-low+1)&1)==1)//val.length is an odd number
        {
            double ev=(val[high]>=0? val[high]:-val[high]);
            if(max<ev) max=ev;
            else if(min>ev) min=ev;
        }
        mm.max=max;
        mm.min=min;
    }
    @Passed
    static <T extends Comparable> void maxMin(T[] val, int low, int high, MaxMin<T> mm)
    {
        T max,min;
        if(val[low].compareTo(val[low+1])>0) {max=val[low];min=val[low+1];}
        else {max=val[low+1];min=val[low];}
        for(int i=2+low;i<high;i+=2)
        {
            if(val[i].compareTo(val[i+1])>0)
            {
                if(max.compareTo(val[i])<0) max=val[i];
                if(min.compareTo(val[i+1])>0) min=val[i+1];
            }
            else
            {
                if(max.compareTo(val[i+1])<0) max=val[i+1];
                if(min.compareTo(val[i])>0) min=val[i];
            }
        }
        if(((high-low+1)&1)==1)//val.length is an odd number
        {
            T ev=val[high];
            if(max.compareTo(ev)<0) max=ev;
            else if(min.compareTo(ev)>0) min=ev;
        }
        mm.max=max;
        mm.min=min;
    }
    //</editor-fold>
    
    public static MaxMin<Integer> maxMinAbs(int[] val) { return maxMinAbs(val, 0, val.length - 1); }
    public static MaxMin<Integer> maxMinAbs(int[] val, int low, int high) {
        MaxMin<Integer> mm = new MaxMin<>();
        Vector.maxMinABS(val, low, high, mm);
        return mm;
    }
    
    public static MaxMin<Integer> maxMinABSIndex(int[] val) {
        MaxMin<Integer> mm = new MaxMin<>();
        Vector.maxMinABSIndex(val, 0, val.length-1, mm);
        return mm;
    }
    
    public static MaxMin<Integer> maxMinABSIndex(int[] val, int low, int high) {
        MaxMin<Integer> mm = new MaxMin<>();
        Vector.maxMinABSIndex(val, low, high, mm);
        return mm;
    }
    
    //<editor-fold defaultstate="collapsed" desc="maxMin<long>">
    public static MaxMin<Long> maxMin(long[] a) { return maxMin(a, 0, a.length - 1); }
    public static MaxMin<Long> maxMin(long[] a, int low, int high) {
        long max, min; 
        if (a[low] > a[low+1]) { max = a[low]; min= a[low+1]; }
        else { max= a[low+1]; min = a[low];}
        for(int i = 2 + low; i < high; i += 2) {
            if(a[i] > a[i + 1]) { if(max < a[i]) max = a[i]; if(min > a[i + 1]) min = a[i + 1]; }
            else { if(max < a[i + 1]) max = a[i + 1]; if(min > a[i]) min=a[i]; }
        }
        if(((high - low + 1) & 1) == 1) {//val.length is an odd number
            long ev = a[high]; 
            if(max < ev) max = ev; else if(min > ev) min = ev;
        }
        return new MaxMin<>(max, min);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="maxMin<int>">
    public static MaxMin<Integer> maxMin(int[] a) { return maxMin(a, 0, a.length - 1); }
    public static MaxMin<Integer> maxMin(int[] a, int low, int high) {
        int max, min; 
        if (a[low] > a[low+1]) { max = a[low]; min= a[low+1]; }
        else { max= a[low+1]; min = a[low];}
        for(int i = 2 + low; i < high; i += 2) {
            if(a[i] > a[i + 1]) { if(max < a[i]) max = a[i]; if(min > a[i + 1]) min = a[i + 1]; }
            else { if(max < a[i + 1]) max = a[i + 1]; if(min > a[i]) min=a[i]; }
        }
        if(((high - low + 1) & 1) == 1) {//val.length is an odd number
            int ev = a[high];
            if(max < ev) max = ev; else if(min > ev) min = ev;
        }
        return new MaxMin<>(max, min);
    }
    
    public static MaxMin<Integer> maxMin(int[] a, int threshold) { return maxMin(a,  0, a.length - 1, threshold); }
    public static MaxMin<Integer> maxMin(int[] a, int low, int high, int threshold) {
        int max, min;
        if (a[low] > a[low+1]) { max = a[low]; min = a[low + 1]; }
        else { max = a[low+1]; min = a[low]; }
        for (int i = 2 + low; i < high; i += 2) {
            if (max - min > threshold) { return null; }//if there exists max-minValue > threshold, return null and early stopping.
            if (a[i] > a[i + 1]) { if(max < a[i]) max = a[i]; if(min > a[i + 1]) min = a[i + 1]; }
            else { if(max < a[i + 1]) max = a[i + 1]; if(min > a[i]) min = a[i]; }
        }
        if (((high - low + 1) & 1) == 1) {//val.length is an odd number
            int ev = a[high];
            if (max < ev) max = ev; else if (min > ev) min=ev;
        }
        return new MaxMin<>(max, min);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="maxMin<short>">
    public static MaxMin<Short> maxMin(short[] a) { return maxMin(a, 0, a.length - 1); }
    public static MaxMin<Short> maxMin(short[] a, int low, int high) {
        short max, min; 
        if (a[low] > a[low+1]) { max = a[low]; min= a[low+1]; }
        else { max= a[low+1]; min = a[low];}
        for(int i = 2 + low; i < high; i += 2) {
            if(a[i] > a[i + 1]) { if(max < a[i]) max = a[i]; if(min > a[i + 1]) min = a[i + 1]; }
            else { if(max < a[i + 1]) max = a[i + 1]; if(min > a[i]) min=a[i]; }
        }
        if(((high - low + 1) & 1) == 1) {//val.length is an odd number
            short ev = a[high];
            if(max < ev) max = ev; else if(min > ev) min = ev;
        }
        return new MaxMin<>(max, min);
    }
    
    public static MaxMin<Short> maxMin(short[] a, int threshold) { return maxMin(a,  0, a.length - 1, threshold); }
    public static MaxMin<Short> maxMin(short[] a, int low, int high, int threshold) {
        short max, min;
        if (a[low] > a[low+1]) { max = a[low]; min = a[low + 1]; }
        else { max = a[low+1]; min = a[low]; }
        for (int i = 2 + low; i < high; i += 2) {
            if (max - min > threshold) { return null; }//if there exists max-minValue > threshold, return null and early stopping.
            if (a[i] > a[i + 1]) { if(max < a[i]) max = a[i]; if(min > a[i + 1]) min = a[i + 1]; }
            else { if(max < a[i + 1]) max = a[i + 1]; if(min > a[i]) min = a[i]; }
        }
        if (((high - low + 1) & 1) == 1) {//val.length is an odd number
            short ev = a[high];
            if (max < ev) max = ev; else if (min > ev) min = ev;
        }
        return new MaxMin<>(max, min);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="maxMin<char>">
    public static MaxMin<Character> maxMin(char[] a) { return maxMin(a, 0, a.length - 1); }
    public static MaxMin<Character> maxMin(char[] a, int low, int high) {
        char max, min; 
        if (a[low] > a[low+1]) { max = a[low]; min= a[low+1]; }
        else { max= a[low+1]; min = a[low];}
        for(int i = 2 + low; i < high; i += 2) {
            if(a[i] > a[i + 1]) { if(max < a[i]) max = a[i]; if(min > a[i + 1]) min = a[i + 1]; }
            else { if(max < a[i + 1]) max = a[i + 1]; if(min > a[i]) min=a[i]; }
        }
        if(((high - low + 1) & 1) == 1) {//val.length is an odd number
            char ev = a[high];
            if(max < ev) max = ev; else if(min > ev) min = ev;
        }
        return new MaxMin<>(max, min);
    }
    
    public static MaxMin<Character> maxMin(char[] a, int threshold) { return maxMin(a,  0, a.length - 1, threshold); }
    public static MaxMin<Character> maxMin(char[] a, int low, int high, int threshold) {
        char max, min;
        if (a[low] > a[low+1]) { max = a[low]; min = a[low + 1]; }
        else { max = a[low+1]; min = a[low]; }
        for (int i = 2 + low; i < high; i += 2) {
            if (max - min > threshold) { return null; }//if there exists max-minValue > threshold, return null and early stopping.
            if (a[i] > a[i + 1]) { if(max < a[i]) max = a[i]; if(min > a[i + 1]) min = a[i + 1]; }
            else { if(max < a[i + 1]) max = a[i + 1]; if(min > a[i]) min = a[i]; }
        }
        if (((high - low + 1) & 1) == 1) {//val.length is an odd number
            char ev = a[high];
            if (max < ev) max = ev; else if (min > ev) min = ev;
        }
        return new MaxMin<>(max, min);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="maxMin<byte>">
    public static MaxMin<Byte> maxMin(byte[] a) { return maxMin(a, 0, a.length - 1); }
    public static MaxMin<Byte> maxMin(byte[] a, int low, int high) {
        byte max, min; 
        if (a[low] > a[low+1]) { max = a[low]; min= a[low+1]; }
        else { max= a[low+1]; min = a[low];}
        for(int i = 2 + low; i < high; i += 2) {
            if(a[i] > a[i + 1]) { if(max < a[i]) max = a[i]; if(min > a[i + 1]) min = a[i + 1]; }
            else { if(max < a[i + 1]) max = a[i + 1]; if(min > a[i]) min=a[i]; }
        }
        if(((high - low + 1) & 1) == 1) {//val.length is an odd number
            byte ev = a[high];
            if(max < ev) max = ev; else if(min > ev) min = ev;
        }
        return new MaxMin<>(max, min);
    }
    
    public static MaxMin<Byte> maxMin(byte[] a, int low, int high, int threshold) {
        byte max, min;
        if (a[low] > a[low+1]) { max = a[low]; min = a[low + 1]; }
        else { max = a[low+1]; min = a[low]; }
        for(int i = 2 + low; i < high; i += 2) {
            if(max - min > threshold) { return null; }//if there exists max-minValue > threshold, return null and early stopping.
            if(a[i] > a[i + 1]) { if(max < a[i]) max = a[i]; if(min > a[i + 1]) min = a[i + 1]; }
            else { if(max < a[i + 1]) max = a[i + 1]; if(min > a[i]) min = a[i]; }
        }
        if(((high - low + 1) & 1) == 1) {//val.length is an odd number
            byte ev = a[high];
            if (max < ev) max = ev; else if (min > ev) min = ev;
        }
        return new MaxMin<>(max, min);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="maxMin<double>">
    public static MaxMin<Double> maxMin(double[] a) { return maxMin(a, 0, a.length - 1); }
    public static MaxMin<Double> maxMin(double[] a, int low, int high) {
        double max, min; 
        if (a[low] > a[low+1]) { max = a[low]; min= a[low+1]; }
        else { max= a[low+1]; min = a[low];}
        for(int i = 2 + low; i < high; i += 2) {
            if(a[i] > a[i + 1]) { if(max < a[i]) max = a[i]; if(min > a[i + 1]) min = a[i + 1]; }
            else { if(max < a[i + 1]) max = a[i + 1]; if(min > a[i]) min=a[i]; }
        }
        if(((high - low + 1) & 1) == 1) {//val.length is an odd number
            double ev = a[high];
            if(max < ev) max = ev; else if(min > ev) min = ev;
        }
        return new MaxMin<>(max, min);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="maxMin<float>">
    public static MaxMin<Float> maxMin(float[] a) { return maxMin(a, 0, a.length - 1); }
    public static MaxMin<Float> maxMin(float[] a, int low, int high) {
        float max, min; 
        if (a[low] > a[low+1]) { max = a[low]; min= a[low+1]; }
        else { max= a[low+1]; min = a[low];}
        for(int i = 2 + low; i < high; i += 2) {
            if(a[i] > a[i + 1]) { if(max < a[i]) max = a[i]; if(min > a[i + 1]) min = a[i + 1]; }
            else { if(max < a[i + 1]) max = a[i + 1]; if(min > a[i]) min=a[i]; }
        }
        if(((high - low + 1) & 1) == 1) {//val.length is an odd number
            float ev = a[high];
            if(max < ev) max = ev; else if(min > ev) min = ev;
        }
        return new MaxMin<>(max, min);
    }
    //</editor-fold>

    public static MaxMin<Double> maxMinABS(double[] val) {
        MaxMin<Double> mm=new MaxMin<>();
        Vector.maxMinABS(val, 0, val.length-1, mm);
        return mm;
    }
    public static MaxMin<Double> maxMinABS(double[] val, int low, int high)
    {
        MaxMin<Double> mm=new MaxMin<>();
        Vector.maxMinABS(val, low, high, mm);
        return mm;
    }
    
    public static <T extends Comparable> MaxMin<T> maxMin(T[] val)
    {
        return Vector.maxMin(val, 0, val.length-1);
    }
    public static <T extends Comparable> MaxMin<T> maxMin(T[] val, int low, int high)
    {
        MaxMin<T> mm=new MaxMin<>();
        Vector.maxMin(val, low, high, mm);
        return mm;
    }
    
    /**
     * <pre>
     * deassign a relative value from 0 to 1, to all element of the input
     * Array {@code val}.
     * 1.find the max and minValue value of {@code val}
     * 2.let {@code base=max-minValue}
     * 2.for each element of val:{@code result[i]=(val[i]-minValue)/base}
     * </pre>
     * @param result
     * @param val 
     */
    @Passed
    public static void relative(double[] result, double[] val)
    {
        MaxMin<Double> mm=Vector.maxMin(val);
        double min=mm.min,base=mm.max-min;
        for(int i=0;i<result.length;i++) result[i]=(val[i]-min)/base;
    }
     /**
     * consider the input Array{@code left}, {@code right} as two points
     * in a multi-dimensional space, find the squre of distance between 
     * the specific two points.
     * @param left the first point 
     * @param right the second point
     * @return distance between the two points
     */
    @Passed
    public static double distanceSquare(double[] left, double[] right)
    {
        double dis=0,r;
        for(int i=0;i<left.length;i++)
            {r=left[i]-right[i];dis+=r*r;}
        return dis;
    }
    /**
     * consider the input Array{@code left}, {@code right} as two points
     * in a multi-dimensional space, cauculate the distance between the 
     * specific two points.
     * @param left the first point 
     * @param right the second point
     * @return distance between the two points
     */
    @Passed
    public static double distance(double[] left, double[] right)
    {
        double dis=0,r;
        for(int i=0;i<left.length;i++)
            {r=left[i]-right[i];dis+=r*r;}
        return Math.sqrt(dis);
    }
    public static double sum(double[] a, int low ,int high)
    {
        double sum=0;
        for(int i=low;i<=high;i++) sum+=a[i];
        return sum;
    }
    public static float sum(float[] a, int low ,int high)
    {
        float sum=0;
        for(int i=low;i<=high;i++) sum+=a[i];
        return sum;
    }
    public static int sum(int[] a, int low ,int high)
    {
        int sum=0;
        for(int i=low;i<=high;i++) sum+=a[i];
        return sum;
    }
    public static double sum(double[] a) {return sum(a, 0, a.length-1);}
    public static float sum(float[] a) {return sum(a, 0, a.length-1);}
    public static int sum(int[] a) {return sum(a, 0, a.length-1);}
    
    public static float straight_quadratic(float[] X, float alpha, float beta, float gamma)
    {
        float v = 0;
        for(int i=0; i < X.length; i++) {
            float x = X[i];
            v += alpha*(x*x) + beta*x + gamma;
        }
        return v;
    }
    
     public static float straight_linear(float[] X, float alpha, float beta)
    {
        float v = 0;
        for(int i=0; i < X.length; i++) {
            float x = X[i];
            v += alpha*x + beta;
        }
        return v;
    }
    
    public static float squareMean(float[] X) {
        float v = 0, alpha = 1.0f / X.length;
        for(int i=0; i<X.length; i++) {
            float x = X[i];
            v += alpha * x*x;
        }
        return v;
    } 
    
    public static float mean(float[] X) {
        float v = 0, alpha = 1.0f / X.length;
        for(int i=0; i<X.length; i++) {
            v += alpha * X[i] ;
        }
        return v;
    }
    
    public static float[] var(float[] X) {
        float mean = mean(X);
        float squareMean = squareMean(X);
        float var = squareMean - mean*mean;
        return new float[]{ var, mean, squareMean };
    }
    
    public static float[] stddev(float[] X) {
        float mean = mean(X);
        float squareMean = squareMean(X);
        float stddev = (float) Math.sqrt(squareMean - mean*mean);
        return new float[]{ stddev, mean, squareMean };
    }
    
    public static float sum(float alpha, float[] a, float beta, int low, int high) {
        float sum=0;
        for(int i=low;i<=high;i++) sum += alpha*a[i] + beta;
        return sum;
    }
    public static float[] sum(float[]... Xs) {
        float[] Y = new float[Xs[0].length];
        for(float[] X : Xs) {
            for(int i=1; i<Y.length; i++) Y[i] += X[i];
        }
        return Y;
    }
    
    
    public static float sum(float alpha, float[] a, float beta) {return sum(alpha, a, beta, 0, a.length-1);}
    
    public static float squareSum(float[] a, int low, int high)
    {
        float sum=0;
        for(int i=low;i<=high;i++) sum+=a[i]*a[i];
        return sum;
    }
    public static float squareSum(float[] a) {return squareSum(a, 0, a.length-1);}
    
    public static float squareSum(float[] a, float alpha, int low, int high)
    {
        float sum=0;
        float k1=alpha*2, k2 = alpha*alpha;
        for(int i=low;i<=high;i++) sum+=a[i]*(a[i]+k1);
        return sum + k2*(high-low+1);
    }
    public static float squareSum(float[] a, float alpha) {return squareSum(a, alpha, 0, a.length-1);}
    
    public static float absSum(float[] a, int low, int high)
    {
        float sum=0;
        for(int i=low;i<=high;i++) sum+=Math.abs(a[i]);
        return sum;
    }
    public static float absSum(float[] a) {return absSum(a, 0, a.length-1);}
    
    public static float absSum(float[] a, float alpha, int low, int high)
    {
        float sum=0;
        for(int i=low;i<=high;i++) sum+=Math.abs(a[i]+alpha);
        return sum;
    }
    public static float absSum(float[] a, float alpha) {return absSum(a, alpha, 0, a.length-1);}
    
    public static long average(long[] arr) { return average(arr, 0, arr.length - 1); }
    public static long average(long[] arr, int low, int high) {
        long avg = 0;
        for(int i = low; i <= high; i++) avg += arr[i];
        return (avg / (high - low + 1));
    }
    
    public static int average(int[] arr) { return average(arr, 0, arr.length - 1); }
    public static int average(int[] arr, int low, int high) {
        long avg = 0;
        for(int i = low; i <= high; i++) avg += arr[i];
        return (int) (avg / (high - low + 1));
    }
    
    public static short average(short[] arr) { return average(arr, 0, arr.length - 1); }
    public static short average(short[] arr, int low, int high) {
        long avg = 0;
        for(int i = low; i <= high; i++) avg += arr[i];
        return (short) (avg / (high - low + 1));
    }
    
    public static byte average(byte[] arr) { return average(arr, 0, arr.length - 1); }
    public static byte average(byte[] arr, int low, int high) {
        long avg = 0;
        for(int i = low; i <= high; i++) avg += arr[i];
        return (byte) (avg / (high - low + 1));
    }
    
    
    public static float average(float[] arr) { return average(arr, 0, arr.length - 1); }
    public static float average(float[] arr, int low, int high) {
        double avg = 0;
        for(int i = low; i <= high; i++) avg += arr[i];
        return (float) (avg / (high - low + 1));
    }
    
    public static double average(double[] arr) { return average(arr, 0, arr.length - 1); }
    public static double average(double[] arr, int low, int high) {
        double avg = 0;
        for(int i = low; i <= high; i++) avg += arr[i];
        return (avg / (high - low + 1));
    }
    
    @Passed
    public static double averageABS(double[] a) {
        double avg=average(a), sum=0, t;
        for(double v:a) {t=v-avg;sum+=(t<0? -t: t);}
        return sum/a.length;
    }
    
    private static double minN(double[] a, int n, int low, int high) {
        //Find the Nth minest factor of the input Array {@code double[] a}.
        if(low>=high) return a[low];
        int mid = Sort.quick_part(a, low, high);
        if(mid==n) return a[mid];
        else if(n<mid) return minN(a, n, low, mid-1);
        else return minN(a, n, mid+1, high);
    }
    
    public static double minN(double[] a, int n, int low, int high, boolean copied)
    {
        if(copied) a=Vector.arrayCopy(a);
        return minN(a, n, low, high);
    }
    public static double minN(double[] a, int n, boolean copied)
    {
        return minN(a, n, 0, a.length-1, copied);
    }
    public static double minN(double[] a, int n)
    {
        return minN(a, n, 0, a.length-1, true);
    }
    public static double maxN(double[] a, int n, int low, int high, boolean copied)
    {
        if(copied) a=Vector.arrayCopy(a);
        return minN(a, a.length-1-n, low, high);
    }
    public static double maxN(double[] a, int n, boolean copied)
    {
        return maxN(a, n, 0, a.length-1, copied);
    }
    public static double maxN(double[] a, int n)
    {
        return minN(a, n, 0, a.length-1, true);
    }
    
    /**
     * find the median of the input Array. It may change the order
     * of elements, if you don't want that, set copied=true, to do this on a
     * new copied Array.
     * @param a
     * @param copied
     * @return 
     */
    @Passed
    public static double median(double[] a, boolean copied)
    {
        if(a.length==0) throw new IAE("Can't find the median in an Array with zero length");
        if(copied) a=Vector.arrayCopy(a);
        double mid1=Vector.minN(a, a.length>>1, 0, a.length-1, false);
        if((a.length&1)==1) return mid1;
        return (mid1+Vector.minN(a, a.length>>1+1, 0, a.length-1, false))/2;
    }
    public static double median(double[] a)
    {
        return median(a, true);
    }
    
    
    @Passed
    public static double variance(int[] val)
    {
       double avg=0,avgs=0;
        for(int i=0;i<val.length;i++)
            {avg+=val[i];avgs+=val[i]*val[i];}
        avg/=val.length;
        avgs/=val.length;
        return avgs-avg*avg;
    }
    @Passed
    public static double variance(long[] val)
    {
       double avg=0,avgs=0;
        for(int i=0;i<val.length;i++)
            {avg+=val[i];avgs+=val[i]*val[i];}
        avg/=val.length;
        avgs/=val.length;
        return avgs-avg*avg;
    }
    @Passed
    public static double variance(double[] val)
    {
        double avg=0,avgs=0;
        for(int i=0;i<val.length;i++)
            {avg+=val[i];avgs+=val[i]*val[i];}
        avg/=val.length;
        avgs/=val.length;
        return avgs-avg*avg;
    }
    public static double stddev(int[] val)
    {
        return Math.sqrt(Vector.variance(val));
    }
    public static double stddev(long[] val)
    {
        return Math.sqrt(Vector.variance(val));
    }
    public static double stddev(double[] val)
    {
        return Math.sqrt(Vector.variance(val));
    }
 
    /**
     * <pre>
     * do normalization on the input Array {@code val}.
     * 1.find the average {@code avg} and the standard derivation {@code stddev} of {@ val}
     * 2.for each element of {@code val}: {@code result[i]=(val[i]-avg)/stddev}
     * </pre>
     * @param result
     * @param val 
     */
    @Passed
    public static void normalize(double[] result, double[] val)
    {
        double avg=0,avgs=0;
        for(int i=0;i<val.length;i++)
            {avg+=val[i];avgs+=val[i]*val[i];}
        avg/=val.length;
        avgs/=val.length;
        avgs=Math.sqrt(avgs);
        for(int i=0;i<val.length;i++)
            result[i]=(val[i]-avg)/avgs;
    }
    
    /**
     * result(1,n)=Vt(1,p)* A(p,n).
     * find the product of between left and each column vector of the input
     * Matrix {@code A}. In linear algebra, we need to transpose V first
     * @param result
     * @param v
     * @param A 
     */
    @Passed
    public static void multiply(double[] result, double[] v, double[][] A)
    {
        for(int j=0,i,width=A[0].length;j<width;j++)
        {
            result[j]=v[0]*A[j][0];
            for(i=1;i<A.length;i++) result[j]+=v[i]*A[j][i];
        }
    }
    /**
     * {@link #multiply(double[], double[], double[][]) }
     * @param v
     * @param A
     * @return 
     */
    public static double[] multiply(double[] v, double[][] A)
    {
        double[] result=new double[A[0].length];
        Vector.multiply(result, v, A);
        return result;
    }
    /**
     * result(n, 1)=At(n,p)*v(p,1).
     * @param result
     * @param v
     * @param A 
     */
    public static void multiply(double[] result, double[][] A, double[] v)
    {
        for(int i=0,j,width=A[0].length;i<v.length;i++)
        {
            result[i]=A[i][0]*v[0];
            for(j=0;j<width;j++) result[i]+=A[i][j]*v[j];
        }
    }
    public static double[] multiply(double[][] A, double[] v)
    {
        double[] result=new double[A.length];
        Vector.multiply(result, A, v);
        return result;
    }
    /**
     * Consider the input Arrays {@code left}, a vectors, find the product
     * of {@left} and the input constant {@code k}.
     * @param result
     * @param left
     * @param k
     */
    @Passed
    public static void multiply(double[] result, double[] left, double k)
    {
        for(int i=0;i<result.length;i++)
            result[i]=left[i]*k;
    }
    @Passed
    public static double dot(double[] A, double[] B)
    {
        double result=0;
        for(int i=0;i<A.length;i++) result+=A[i]*B[i];
        return result;
    }
    public static float dot(float[] A, float[] B)
    {
        float result=0;
        for(int i=0;i<A.length;i++) result+=A[i]*B[i];
        return result;
    }
    /**
     * Consider the two input Arrays {@code left}, {@code right} as two 
     * vectors, find the cosine of the Angle between the two vectors, as 
     * the Angle belongs to [0, Pi].
     * @param left
     * @param right
     * @return 
     */
    @Passed
    public static double cosAngle(double[] left, double[] right)
    {
        double product=0,sub,mod=0;
        for(int i=0;i<left.length;i++)
        {
            product+=left[i]*right[i];
            sub=right[i]-left[i];
            mod+=sub*sub;
        }
        return product/mod;
    }
    /**
     * Consider the two input Arrays {@code left}, {@code right} as two 
     * vectors, find the sine of the Angle between the two vectors, as 
     * the Angle belongs to [0, Pi].
     * @param left
     * @param right
     * @return 
     */
    @Passed
    public static double sinAngle(double[] left, double[] right)
    {
        double cos=Vector.cosAngle(left, right);
        return Math.sqrt(1-cos*cos);
    }
    
    public static double[] posibility(int[] f)
    {
        double[] p=new double[f.length];
        int num=0;
        for(int i=0;i<f.length;i++) num+=f[i];
        for(int i=0;i<p.length;i++) p[i]=f[i]*1.0/num;
        return p;
    }
    public static double[][] posibility(int[][] fs)
    {
        double[][] ps=new double[fs.length][];
        for(int i=0;i<ps.length;i++) ps[i]=posibility(fs[i]);
        return ps;
    }
    public static double gini(double[] p)
    {
        double result=p[0]*p[0];
        for(int i=1;i<p.length;i++) result+=p[i]*p[i];
        return 1-result;
    }
    
    public static double entropyE(double[] p) {
        double result=0;
        for(int i=0;i<p.length;i++) result+=(p[i]==0? 0:p[i]*Math.log(p[i]));
        return -result;
    }
    
    public static float entropyE(float[] p) {
        float result = 0;
        for(int i=0; i<p.length; i++) result += (p[i]==0? 0 : p[i] * Math.log(p[i]));
        return -result;
    }
    
    public static double entropyE(int[] f) {return entropyE(posibility(f));}
    public static double entropy2(double[] p) {return entropyE(p)/Math.log(2);}
    public static double entropy2(int[] f) {return entropyE(f)/Math.log(2);}
    
    private static double InfoEntropyE(double[][] p, int[] nums)
    {
        double[] infos=new double[p.length];
        int num=0;
        for(int i=0;i<infos.length;i++) {infos[i]=entropyE(p[i]);num+=nums[i];}
        
        double info=0;
        for(int i=0;i<infos.length;i++) {info+=infos[i]*nums[i]/num;}
        return info;
    }
    public static double InfoEntropyE(int[][] f) 
    {
        int[] nums=new int[f.length];
        for(int i=0;i<f.length;i++) nums[i]=sum(f[i]);
        return InfoEntropyE(posibility(f), nums);
    }
    public static double InfoEntropy2(int[][] f) {return InfoEntropyE(f)/Math.log(2);}
      /**
     * find the reciprocal for each component of the input Array {@code val}.
     * @param result
     * @param val
     */
    @Passed
    public static void reciprocal(double[] result, double[] val)
    {
        for(int i=0;i<result.length;i++) result[i]=1/val[i];
    }
    @Passed
    public static void relu(double[] result, double[] val)
    {
        for(int i=0;i<result.length;i++)
            result[i]=(val[i]>0? val[i]:0);
    }
    
    public static void sigmoid(double[] Y) {
        for(int i=0; i < Y.length; i++)  Y[i] = 1 / (1 + Math.exp(-Y[i]));
    }
    
    @Passed
    public static void sigmoid(double[] X, double[] Y)
    {
        for(int i=0;i<Y.length;i++)
            Y[i]=1/(1+Math.exp(-X[i]));
    }
    @Passed
    public static void unSigmoid(double[] result)
    {
        for(int i=0;i<result.length;i++)
            result[i]=-Math.log(1/result[i]-1);
    }
    @Passed
    public static void unSigmoid(double[] result, double[] val)
    {
        for(int i=0;i<result.length;i++)
            result[i]=-Math.log(1/val[i]-1);
    }
    @Passed
    public static void tanh(double[] result)
    {
        for(int i=0;i<result.length;i++)
            result[i] = 1 - 2/(Math.exp(2*result[i])+1);
    }
    @Passed
    public static void tanh(double[] result, double[] val)
    {
        for(int i=0;i<result.length;i++)
            result[i]=1-2/(Math.exp(2*val[i])+1);
    }
    @Passed
    public static void unTanh(double[] result)
    {
        for(int i=0;i<result.length;i++)
            result[i]=-0.5*Math.log(1/(1-result[i])-1);
    }
    @Passed
    public static void unTanh(double[] result, double[] val)
    {
        for(int i=0;i<result.length;i++)
            result[i]=-0.5*Math.log(1/(1-val[i])-1);
    }
    @Passed
    public static void softPlus(double[] result, double[] val)
    {
        for(int i=0;i<result.length;i++)
            result[i]=Math.log(Math.exp(val[i]+1));
    }
    @Passed
    public static void unSoftPlus(double[] result, double[] val)
    {
        for(int i=0;i<result.length;i++)
            result[i]=Math.log(Math.exp(val[i]-1));
    }
    @Passed
    public static double norm(double[] x)
    {
        double sum=0;
        for(double v:x) sum+=v*v;
        return Math.sqrt(sum);
    }
    @Passed
    public static double normSquare(double[] x)
    {
        double sum=0;
        for(double v:x) sum+=v*v;
        return sum;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Vector: math function">
    //<editor-fold defaultstate="collapsed" desc="math: trigonometric Function">
    public static float[] sin(float[] X) {
        float[] Y = new float[X.length];
        for (int i=0; i<X.length; i++) Y[i] = (float) Math.sin(X[i]);
        return Y;
    }
    public static float[] sin(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.sin(alpha*X[i] + beta);
        return Y;
    }
    
    public static float[] cos(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.cos(X[i]);
        return Y;
    }
    public static float[] cos(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.cos(alpha*X[i] + beta);
        return Y;
    }
    
    public static float[] sec(float[] X, float alpha, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++)
            Y[i] = (float) (1.0 / Math.cos(alpha*X[i] + beta));
        return Y;
    }
    
    public static float[] sec_deri(float[] X, float alpha, float beta) {
        float[] deriY = new float[X.length];
        for(int i=0; i<X.length; i++) {
            float x = alpha*X[i] + beta;
            float y = (float) Math.cos(x);
            deriY[i] = (float) (alpha * Math.sin(x) / (y*y));
        }
        return deriY;
    }
    
    public static float[] csc(float[] X, float alpha, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) 
            Y[i] = (float) (1.0 / Math.sin(alpha*X[i] + beta));
        return Y;
    }
    
    public static float[] csc_deri(float[] X, float alpha, float beta) {
        float[] deriY = new float[X.length];
        for(int i=0; i<X.length; i++) {
            float x = alpha*X[i] + beta;
            float y = (float) Math.sin(x);
            deriY[i] = (float) (-alpha*Math.cos(x) / (y*y));
        }
        return deriY;
    }
    
    public static void halfSin(float amp, float alpha, float[] X, float beta, float[] Y, int length) {
        for(int i=0; i<length; i++) {
            float x = alpha*X[i] + beta;
            x -= Math.floor(x/Math.PI + 0.5f) * Math.PI;
            Y[i] = amp * (float) Math.sin(x);
        }
    }
    public static void halfSin_Deri(float alpha, float[] Y, float[] deriY, int length) {
        for(int i=0; i<length; i++) {
            float y = Y[i];
            deriY[i] = (float) (alpha * Math.sqrt(1 - y*y));
        }
    }
    
    public static float[] equal(float[] X1, float[] X2, float min, float max) {
        float[] Y = new float[X1.length];
        for(int i=0; i<Y.length; i++) {
            float div = Math.abs(X1[i] - X2[i]);
            boolean flag = div <= max && div >= min;
            Y[i] = flag? 1 : 0;
        }
        return Y;
    }
    
    public static float[] equal(byte[] X1, byte[] X2, byte min, byte max) {
        float[] Y = new float[X1.length];
        for(int i=0; i<Y.length; i++) {
            int div = Math.abs(X1[i] - X2[i]);
            boolean flag = div <= max && div >= min;
            Y[i] = flag? 1 : 0;
        }
        return Y;
    }

    public static float[] equal(int[] X1, int[] X2, int min, int max) {
        float[] Y = new float[X1.length];
        for(int i=0; i<Y.length; i++) {
            int div = Math.abs(X1[i] - X2[i]);
            boolean flag = div <= max && div >= min;
            Y[i] = flag? 1 : 0;
        }
        return Y;
    }

    public static float[] tan(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.tan(X[i]);
        return Y;
    }        
    public static float[] tan(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.tan(alpha*X[i] + beta);
        return Y;
    }   
    public static float[] tan_deri(float alpha, float[] X, float beta) {
        float[] deriY = new float[X.length];
        for(int i=0; i<X.length; i++) {
            float x = alpha*X[i] + beta;
            float cos_x = (float) Math.cos(x);
            deriY[i] = alpha / (cos_x * cos_x);
        }
        return deriY;
    }
    
    public static float[] cot(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) (1.0f / Math.tan(X[i]));
        return Y;
    }
    public static float[] cot(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) (1/Math.tan(alpha*X[i] + beta));
        return Y;
    }
    public static float[] cot_deri(float alpha, float[] X, float beta) {
        float[] deriY = new float[X.length];
        for(int i=0; i<X.length; i++) {
            float x = alpha*X[i] + beta;
            float sin_x = (float) Math.sin(x);
            deriY[i] = -alpha / (sin_x * sin_x);
        }
        return deriY;
    }
    
    public static float[] arcsin(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.asin(X[i]);
        return Y;
    }
    public static float[] arcsin(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.asin(alpha * X[i] + beta);
        return Y;
    }
    public static float[] asin_Deri(float alpha, float[] X, float beta) {
        float[] deriY = new float[X.length];
        for(int i=0; i<X.length; i++) {
            float x = alpha*X[i] + beta;
            deriY[i] = (float) (alpha / Math.sqrt(1 - x*x));
        }
        return deriY;
    }
  
    public static float[] arccos(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.acos(X[i]);
        return Y;
    }
    public static float[] arccos(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.acos((alpha * X[i] + beta));
        return Y;
    }
    
    public static float[] arctan(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.atan(X[i]);
        return Y;
    }        
    public static float[] arctan(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) (Math.atan(alpha*X[i] + beta));
        return Y;
    }     
    public static float[] atan_deri(float alpha, float[] X, float beta) {
        float[] deriY = new float[X.length];
        for(int i=0; i<X.length; i++) {
            float x = alpha*X[i] + beta;
            deriY[i] = alpha / (1 + x*x);
        }
        return deriY;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="math: exponential functions">
    public static float[] sqrt(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.sqrt(X[i]);
        return Y;
    }
    public static float[] sqrt(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.sqrt(alpha*X[i] + beta);
        return Y;
    }
    public static float[] sqrt_deri(float alpha, float[] X, float beta) { 
        float[] deriY = new float[X.length];
        for(int i=0; i<X.length; i++) {
            float x = (float) Math.sqrt(alpha * X[i] + beta);
            deriY[i] =  alpha / (2 * x);
        }
        return deriY;
    }
    public static float[] sqrt_deri(float[] Y, float alpha) {
        float[] deriY = new float[Y.length];
        for(int i=0; i<Y.length; i++) deriY[i] = 0.5f * alpha / Y[i];
        return deriY;
    }
    
    public static float[] sqrt_quadratic2(float[] X1, float[] X2, 
            float k11, float k12, float k22, 
            float k1, float k2, float C) 
    {
        float[] Y = new float[X1.length];
        for(int i=0; i<Y.length; i++) {
            float x1 = X1[i], x2 = X2[i];
            float y = k11*x1*x1 + k12*x1*x2 + k22*x2*x2 + k1*x1 + k2*x2 + C;
            Y[i] = (float) Math.sqrt(y);
        }
        return Y;
    }
    
    public static float[] linear_greater(float alpha, float[] X, float  beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<Y.length; i++) {
            if(alpha * X[i] + beta > 0) Y[i] = 1;
            else Y[i] = 0;
        }
        return Y;
    }
    
    public static float[] linear_greater2(float[] X1, float[] X2, float alpha, float beta, float gamma) {
        float[] Y = new float[X1.length];
        for(int i=0; i<Y.length; i++) {
            if(alpha*X1[i] + beta*X2[i] + gamma > 0) Y[i] = 1;
            else Y[i] = 0;
        }
        return Y;
    }
    
    public static float[] linear_greater_switch(float alpha, float[] X, float beta, float v1, float v2) {
        float[] Y = new float[X.length];
        for(int i=0; i<Y.length; i++) {
            if(alpha * X[i] + beta > 0) Y[i] = v1;
            else Y[i] = v2;
        }
        return Y;
    }
    
    public static float[] linear_greater_switch_mul(
            float alpha, float[] X1, float beta, 
            float[] X2, float v1, float v2) 
    {
        float[] Y = new float[X1.length];
        for(int i=0; i<Y.length; i++) {
            if(alpha * X1[i] + beta > 0) Y[i] = v1;
            else Y[i] = v2;
            Y[i] *= X2[i];
        }
        return Y;
    }
    
    public static float[] linear_bound_switch_mul(
            float alpha, float[] X1, float vmin, float vmax, 
            float[] X2, float v1, float v2, float v3)
    {
        float[] Y = new float[X1.length];
        if(vmin > vmax) { float t = vmin; vmin = vmax; vmax = t; }
        for(int i=0; i<Y.length; i++) {
            float v = alpha * X1[i];
            if(v >= vmax)      Y[i] = v3;
            else if(v <= vmin) Y[i] = v1;
            else               Y[i] = v2;
            Y[i] *= X2[i];
        }
        return Y;
    }
    
    public static float[] exp(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.exp(X[i]);
        return Y;
    } 
    public static float[] exp(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.exp(alpha*X[i] + beta);
        return Y;
    }
    
    public static float[] log(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.log(X[i]);
        return Y;
    }
    public static float[] log(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.log(alpha * X[i] + beta);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="math: hyperbolic functions">
    public static float[] sinh(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) {
            float expx = (float) Math.exp(X[i]);
            Y[i] = (expx - 1 / expx) / 2;
        }
        return Y;
    }
    public static float[] sinh(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) {
            float expx = (float) Math.exp(alpha*X[i] + beta);
            Y[i] = (expx - 1 / expx) / 2;
        }
        return Y;
    }
    
    public static float[] cosh(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++)  {
            float expx = (float) Math.exp(X[i]);
            Y[i] = (expx + 1 / expx) / 2;
        }
        return Y;
    }
    public static float[] cosh(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) {
            float expx = (float) Math.exp(alpha*X[i] + beta);
            Y[i] = (expx + 1 / expx) / 2;
        }
        return Y;
    }
    
    public static float[] tanh(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) {
            float expX = (float) Math.exp(X[i]);
            Y[i] = (expX - 1/expX) / (expX + 1/expX);
        }
        return Y;
    }
    public static float[] tanh_deri(float[] Y) {
        float[] deriY = new float[Y.length];
        for(int i=0; i<Y.length; i++) deriY[i] = 1 - Y[i]*Y[i];
        return deriY;
    }
   
    public static float[] sigmoid(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++)
            Y[i] = (float) (1 / (1 + Math.exp(-X[i])));
        return Y;
    }
    public static void sigmoid_Deri(float[] Y, float[] deriY, int length) {
        for(int i=0; i<length; i++)
            deriY[i] = Y[i] * (1 - Y[i]);
    }
    
    public static float[] softmax(float[] X) {
        float[] Y = new float[X.length];
        float sum = 0, maxX = Vector.maxValue(X);
        for(int i=0; i<X.length; i++) {
            Y[i] = (float) Math.exp(X[i] - maxX);
            sum += Y[i];
        }
        for(int i=0; i<Y.length; i++) Y[i] /= sum;
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Basic Function">
    public static float[] sadd(float[] X, float C) { return linear(1.0f, X, C); }
    public static float[] ssub(float[] X, float C) { return linear(1.0f, X, -C); }
    public static float[] smul(float[] X, float C) { return linear(C, X, 0.0f); }
    public static float[] sdiv(float[] X, float C) { return linear(1.0f / C, X, 0.0f); }
    public static float[] linear(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = alpha*X[i] + beta;
        return Y;
    }
    
    public static float[][][] linear_center(float[][][] X1, float[][] X2, 
            float alpha, float beta, float gamma) 
    {
        int dim0 = X1.length;
        int dim1 = X1[0].length;
        int dim2 = X1[0][0].length;
        float[][][] Y = new float[dim0][dim1][dim2];
        
        for(int d0=0; d0<dim0; d0++)
        for(int d1=0; d1<dim1; d1++)
        for(int d2=0; d2<dim2; d2++) {
            float x2 = X2[d0][d2];
            float x1 = X1[d0][d1][d2];
            Y[d0][d1][d2] = alpha*x1 + beta*x2 + gamma;
        }
        return Y;
    }
    
    public static float[] rpl(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = 1.0f / X[i];
        return Y;
    }
    public static float[] rpl(float alpha, float[] X, float beta, float gamma) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = alpha / (X[i] + beta) + gamma;
        return Y;
    }
    public static float[] rpl_deri(float[] X, float alpha, float beta){
        float[] deriY = new float[X.length];
        for(int i=0; i<X.length; i++) {
            float x = X[i] + beta;
            deriY[i] = -alpha / (x * x);
        }
        return deriY;
    }
    
    public static float[] div(
            float alpha1, float[] X1, float beta1,
            float alpha2, float[] X2, float beta2, 
            float gamma)  {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) {
            float x1 = alpha1*X1[i] + beta1;
            float x2 = alpha2*X2[i] + beta2;
            Y[i] = x1 / x2 + gamma;
        }
        return Y;
    }
    public static void div_Deri(float[] deriX1, float[] deriX2,
            float[] X1, float alpha1, float beta1,
            float[] X2, float alpha2, float beta2,
            int length) {
        for(int i=0; i<length; i++) {
            float x1 = X1[i];
            float x2 = X2[i];
            deriX1[i] = alpha1 / (alpha2*x2 + beta2);
            deriX2[i] = -alpha2 * (alpha1*x1 + beta1) /(alpha2*x2 + beta2)/ (alpha2*x2 + beta2);
        }
    }
    
     public static float[] mul_squareDiv(
            float alpha1, float[] X1, float beta1,
            float alpha2, float[] X2, float beta2, 
            float alpha3, float[] X3, float beta3, 
            float gamma)
    {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) {
            float x1 = alpha1*X1[i] + beta1;
            float x2 = alpha2*X2[i] + beta2;
            float x3 = alpha3*X3[i] + beta3;
            Y[i] = (x1 * x2) / (x3 * x3) + gamma;
        }
        return Y;
    }
    
    //<editor-fold defaultstate="collapsed" desc="linear2">
    public static float[] add(float[] X1, float[] X2) {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) Y[i] = X1[i] + X2[i];
        return Y;
    }
    
    public static int[] sub(int[] X1, int[] X2) {
        int[] Y = new int[X1.length];
        for(int i=0; i<X1.length; i++) Y[i] = X1[i] - X2[i];
        return Y;
    }
    public static float[] sub(float[] X1, float[] X2) {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) Y[i] = X1[i] - X2[i];
        return Y;
    }
    
    
    public static float[] add(float alpha, float[] X1, float beta, float[] X2) {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) Y[i] = alpha*X1[i] + beta*X2[i];
        return Y;
    }
    
    public static float[] linear2(float[] X1, float[] X2, float alpha, float beta, float gamma) {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) Y[i] = alpha*X1[i] + beta*X2[i] + gamma;
        return Y;
    }
    
    public static float[] mul_linear2(float[] X, float[] X1, float[] X2, float alpha, float beta, float gamma) {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) Y[i] = X[i] * (alpha*X1[i] + beta*X2[i] + gamma);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="quadratic2">
    public static float[] mul(float[] X1, float[] X2) {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) Y[i] = X1[i] * X2[i];
        return Y;
    }
    public static float[] mul(float alpha, float[] X1, float[] X2) {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) Y[i] = alpha * X1[i] * X2[i];
        return Y;
    }
    public static float[] squareAdd(float[] X1, float[] X2) {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) {
            float x1 = X1[i];
            float x2 = X2[i];
            Y[i] = x1*x1 + x2*x2;
        }
        return Y;
    }
    public static float[] squareSub(float[] X1, float[] X2) {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) {
            float x1 = X1[i];
            float x2 = X2[i];
            Y[i] = x1*x1 - x2*x2;
        }
        return Y;
    }
    public static float[] squareAdd(float alpha, float[] X1, float beta, float[] X2) {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) {
            float x1 = X1[i];
            float x2 = X2[i];
            Y[i] = alpha*x1*x1 + beta*x2*x2;
        }
        return Y;
    }
    
    public static float[] quadratic2(float[] X1, float[] X2,
            float k11, float k12, float k22,
            float k1, float k2, float C)
    {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) {
            float x1 = X1[i];
            float x2 = X2[i];
            Y[i] = k11*x1*x1 + k12*x1*x2 + k22*x2*x2 + k1*x1 + k2*x2 + C;
        }
        return Y;
    }
    
    public static float[][][] quadratic2_center(float[][][] X1, float[][] X2,
            float k11, float k12, float k22, float k1, float k2, float C)
    {
        int dim0 = X1.length;
        int dim1 = X1[0].length;
        int dim2 = X1[0][0].length;
        float[][][] Y = new float[dim0][dim1][dim2];
        
        for(int d0=0; d0<dim0; d0++)
        for(int d1=0; d1<dim1; d1++)
        for(int d2=0; d2<dim2; d2++) {
            float x2 = X2[d0][d2];
            float x1 = X1[d0][d1][d2];
            Y[d0][d1][d2] = k11*x1*x1 + k12*x1*x2 + k22*x2*x2 + k1*x1 + k2*x2 + C;
        }
        return Y;
    }
    
    
    public static void binomial_Deri(
            float[] deriX1, float[] deriX2,
            float[] X1, float[] X2,
            float k11, float k12, float k22, 
            float k1, float k2,
            int length)
    {
        for(int i=0; i<length; i++) {
            float x1 = X1[i];
            float x2 = X2[i];
            deriX1[i] = 2*k11*x1 + k12 * x2 + k1;
            deriX2[i] = 2*k22*x2 + k12 * x1 + k2;
        }
    }
    //</editor-fold>
    
    public static float[] linear_int8_to_float(float alpha, byte[] BX, float beta) {
        float[] X = new float[BX.length];
        for(int i=0; i<X.length; i++) X[i] = alpha * BX[i] + beta;
        return X;
    }
    
    public static byte[] linear_float_to_int8(float alpha, float[] X, float beta) {
        byte[] BX = new byte[X.length];
        for(int i=0; i<BX.length; i++) BX[i] = (byte) (alpha * X[i] + beta);
        return BX;
    }
    
    public static float[] pix2tensor(byte[] pixel) {
        float[] X = new float[pixel.length];
        for(int i=0; i<X.length; i++) {
            int pix = ((int)pixel[i]) & 0xff;
            X[i] = pix / 255.0f;
        }
        return X;
    }
    
    public static byte[] tensor2pix(float[] X) {
        byte[] pixel = new byte[X.length];
        for(int i=0; i<X.length; i++) {
            float x = X[i];
            if(x > 1) x = 1;
            else if(x < 0) x = 0;
            pixel[i] = (byte) (x * 255.0f);
        }
        return pixel;
    }
    
    public static float[] abs(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = Math.abs(X[i]);
        return Y;
    }
    public static float[] abs(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = Math.abs(alpha*X[i] + beta);
        return Y;
    }
    
    public static void sign(float[] X, float[] Y, int length) {
        for(int i=0;i<length;i++)  {
            if(X[i]>0) Y[i] = 1;
            else if(X[i] == 0) Y[i] = 0;
            else Y[i] = -1;
        }
    }
    public static void sign(float alpha, float[] X, float beta, float[] Y, int length) {
        for(int i=0; i<length; i++)  {
            float x = alpha*X[i]+beta;
            if(x>0) Y[i] = 1;
            else if(x == 0) Y[i]=0;
            else Y[i] = -1;
        }
    }
    
    public static float[] square(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) {
            float x = X[i];
            Y[i] = x * x;
        }
        return Y;
    }
    public static float[] square(float alpha, float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) {
            float x = X[i];
            Y[i] = alpha * x * x;
        }
        return Y;
    }
    public static float[] quadratic(float[] X, float alpha, float beta, float gamma) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) {
            float x = X[i];
            Y[i] = alpha*x*x +  beta*x + gamma;
        }
        return Y;
    }
    
    public static void quadratic_Deri(float[] X, float alpha, float beta, float[] deriY, int length){
        System.out.println(alpha + ":" + beta);
        for(int i=0; i<length; i++) {
            float x = X[i];
            deriY[i] = 2*alpha*x + beta;
        }
    }
    
    public static void ceil(float[] X, float[] Y, int length) {
        for(int i=0;i<length;i++) Y[i]=(float) Math.ceil(X[i]);
    }
    public static void ceil(float alpha, float[] X, float beta, float[] Y, int length) {
        for(int i=0;i<length;i++) Y[i]=(float) Math.ceil(alpha*X[i] + beta);
    }
    
    public static void floor(float[] X, float[] Y, int length)
    {
        for(int i=0;i<length;i++) Y[i]=(float) Math.floor(X[i]);
    }
    public static void floor(float alpha, float[] X, float beta, float[] Y, int length)
    {
        for(int i=0;i<length;i++) Y[i]=(float) Math.floor(alpha*X[i] + beta);
    }
    
    public static void max(float[] X, float vmax, float[] Y, int length)
    {
        for(int i=0;i<length;i++) Y[i] = Math.max(X[i], vmax);
    }
    
    public static void min(float[] X, float vmin, float[] Y, int length)
    {
        for(int i=0;i<length;i++) Y[i] = Math.min(X[i], vmin);
    }
    
    public static void clip(float[] X, float min, float max, float[] Y, int length)
    {
        if(min > max) {float t = min; min = max; max = t;}
        for(int i=0;i<length;i++) 
        {
            if(X[i] > max) Y[i] = max;
            else if(X[i] > min) Y[i] = X[i];
            else Y[i] = min;
        }
    }
   
    public static float[] relu(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = Math.max(X[i], 0);
        return Y;
    }
    
    public static float[] relu_deri(float[] Y) {
        float[] deriY = new float[Y.length];
        for(int i=0; i<Y.length; i++) {
            if(Y[i] > 0) deriY[i] = 1.0f;
            else deriY[i] = 0;
        }
        return deriY;
    }
    
    public static float[] leakyRelu(float[] X, float k) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) {
            if(X[i] > 0) Y[i] = X[i];
            else Y[i] = k*X[i];
        }
        return Y;
    }
    
    public static void leakyRelu_Deri_X(float[] X, float k, float[] deriY, int length) {
        for(int i=0; i<length; i++)
            deriY[i] = (X[i] > 0 ? 1.0f : k);
    }
    
    public static float[] leakyRelu_Deri(float[] Y, float k) {
        float[] deriY = new float[Y.length];
        for(int i=0; i<Y.length; i++) deriY[i] = (Y[i] > 0 ? 1.0f : k);
        return deriY;
    }
    
    public static float[] gelu(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) {
            float x = X[i];
            float u = -1.5957692f * x * (1.0f + 0.044715f * x * x);
            Y[i] = (float) (x / (1.0f + Math.exp(u)));
        }
        return Y;
    }
        
    public static float[] gelu_dei(float[] X) {
        float[] deriY = new float[X.length];
        for(int i=0; i<X.length; i++) {
            float x = X[i];
            float u = -1.5957692f * x * (1.0f + 0.044715f * x * x);
            float expu = (float) Math.exp(u);
            float B = 1.0f / (expu + 1.0f);
            float A = 1.0f - (expu * B) * (u - 0.14270963f * x * x);
            deriY[i] = A * B;
        }
        return deriY;
    }    
    
    
    public static void sin_Deri(float[] X, float alpha, float beta, float[] deriY, int length){
        for(int i=0; i<length; i++)
            deriY[i] = (float) (alpha * Math.cos(alpha * X[i] + beta));
    }
    
    public static void abs_Deri(float[] X, float alpha, float[] deriY, int length) {
        for(int i=0; i<length; i++){
            float x = X[i];
            if(x > 0) deriY[i] = alpha;
            else if(x < 0 ) deriY[i] = -alpha;
            else deriY[i] = 0;
        }
    }
    
    public static float[] elu(float[] X, float alpha, float k) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) {
            if(X[i] > 0) Y[i] = X[i];
            else Y[i]= k * (float) (Math.exp(X[i]) - 1.0f);
            Y[i] *= alpha;
        }
        return Y;
    }
    
    public static void elu_Deri(float[] Y, float alpha, float beta, float[] deriY, int length)
    {
        for(int i=0;i<length;i++) {
            if(Y[i] > 0) deriY[i] = alpha;
            else deriY[i] = Y[i] + alpha*beta;
        }
    }
    public static void elu_Deri_2(float[] X, float alpha, float beta, float[] R, int length)
    {
        for(int i=0;i<length;i++)
        {
            if(X[i]>0) R[i] = 1;
            else R[i] = beta * (float)Math.exp(X[i]);
            R[i] *= alpha;
        }
    }
    
    public static float[] softplus(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) 
            Y[i] = (float) Math.log1p(Math.exp(X[i]));
        return Y;
    }
    public static void softPlus_Deri(float[] Y, float[] deriY, int length)
    {
        for(int i=0;i<length;i++)
            deriY[i] = 1 - (float) Math.exp(-Y[i]);
    }
    public static void softPlus_Deri_2(float[] X, float[] R, int length)
    {
        for(int i=0;i<length;i++)
            R[i] = 1 - 1/(1 + (float)Math.exp(X[i]));
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Logarithm">
    public static final float log2f = (float) Math.log(2);
    public static float[] log2(float[] X) {
        float[] Y = new float[X.length];
        for (int i=0; i<X.length; i++) Y[i] = (float) Math.log(X[i]) / log2f;
        return Y;
    } 
    public static float[] log2(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for (int i=0; i<X.length; i++) Y[i] = (float) Math.log(alpha*X[i] + beta) / log2f;
        return Y;
    } 
    
    public static float[] log10(float[] X) {
        float[] Y = new float[X.length];
        for (int i=0; i<X.length; i++) Y[i] = (float) Math.log10(X[i]);
        return Y;
    }
    public static float[] log10(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i]=(float) Math.log10(alpha*X[i] + beta);
        return Y;
    }
    
    public static float[] log(float v, float[] X) {
        float logv = (float) Math.log(v);
        float[] Y = new float[X.length];
        for (int i=0;i<X.length; i++) Y[i] = (float) (Math.log(X[i]) / logv);
        return Y;
    }
    public static float[] log(float v, float alpha, float[] X, float beta) {
        float logv = (float) Math.log(v);
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) (Math.log(alpha*X[i] + beta) / logv);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Difference & Pertinence">
    public static void L1(float[] Yh, float[] Y, float[] L) {
        for(int i=0; i<L.length; i++) L[i] = Math.abs(Yh[i] - Y[i]);
    }
    public static void L1_deltaYh(float[] Yh, float[] Y, float[] deltaYh) {
        for(int i=0;i<deltaYh.length;i++)
            deltaYh[i] = Math.signum(Yh[i] - Y[i]);
    }
      
    public static void L2(float[] Yh, float[] Y, float[] L) {
        for(int i=0;i<L.length;i++) {
            float div = Yh[i]-Y[i];
            L[i] = 0.5f*div*div;
        }
    }
    public static void L2_deltaYh(float[] Yh, float[] Y, float[] deltaYh) {
        for(int i=0; i<deltaYh.length; i++) 
            deltaYh[i] = Yh[i] - Y[i];
    }
    
    public static void smoothL1(float[] Yh, float[] Y, float[] L) {
        for(int i=0; i< L.length; i++) {
            float div = Math.abs(Yh[i] - Y[i]);
            if(div <= 1) L[i] = 0.5f * div * div;
            else L[i] = div - 0.5f;
        }
    }
    public static void smoothL1_deltaYh(float[] Yh, float[] Y, float[] deltaYh) {
        for(int i=0; i<deltaYh.length; i++) {
            float div = Math.abs(Yh[i] - Y[i]);
            if(div <= 1) deltaYh[i] = Yh[i] - Y[i];
            else deltaYh[i] = Math.signum(Yh[i] - Y[i]);
        }
    }
    
    public static void crossEntropy(float[] Yh, float[] Y, float[] L) {
        for(int i=0;i<L.length;i++)
            L[i] = (float) (-Y[i] * Math.log(Yh[i]) 
                    + (Y[i] - 1) * Math.log(1 - Yh[i]));
    }
    public static void crossEntropy_deltaYh(float[] Yh, float[] Y, float[] deltaYh) {
        for(int i=0; i<deltaYh.length; i++)
            deltaYh[i] = (-Y[i] / Yh[i] + (Y[i] - 1) / (Yh[i] - 1));
    }
    
    public static void balancedCrossEntropy(float[] Yh, float[] Y, float alpha, float beta, float[] L) {
        for(int i=0;i<L.length;i++)
            L[i] = (float) (-alpha * Y[i] * Math.log(Yh[i])
                    + beta * (Y[i] - 1) * Math.log(1 - Yh[i]));
    }
    public static void balancedCrossEntropy_deltaYh(float[] Yh, float[] Y, float alpha, float beta, float[] deltaYh) {
        for(int i=0; i<deltaYh.length; i++)
            deltaYh[i] = -alpha * (Y[i] / Yh[i]) + beta * (Y[i] - 1) / (Yh[i] - 1);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Vector-math-Function">
    public static int mul(int[] a) {return multiple(a, 0, a.length-1);}
    public static int multiple(int[] a, int start, int end) {
        int mul = 1;
        for(int i=start; i<=end; i++) mul *= a[i];
        return mul;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="math: image functions">
    public static byte[] img_linear(float alpha, byte[] X, float beta) {
        byte[] Y = new byte[X.length];
        for(int i=0; i<X.length; i++) {
            int x = X[i] & 0xff;
            float y = alpha*x + beta; 
            if(y > 255) y = 255; else if(y < 0) y = 0;
            Y[i] = (byte) y;
        }
        return Y;
    }
            
    public static byte[] img_log(float C, float alpha, byte[] X, float beta) {
        byte[] Y = new byte[X.length];
        for(int i=0; i<X.length; i++) {
            int x = X[i] & 0xff;
            float y = (float) (C*Math.log(alpha*x + beta)); 
            if(y > 255) y = 255; else if(y < 0) y = 0;
            Y[i] = (byte) y;
        }
        return Y;
    }
    
    public static byte[] img_exp(float alpha, byte[] X, float beta, float C) {
        byte[] Y = new byte[X.length];
        for(int i=0; i<X.length; i++) {
            int x = X[i] & 0xff;
            float y = (float) (Math.exp(alpha*x + beta) + C); 
            if(y > 255) y = 255; else if(y < 0) y = 0;
            Y[i] = (byte) y;
        }
        return Y;
    }
    
    public static byte[] img_linear2_row(byte[] X1, float[] X2,
            float alpha, float beta, float gamma,
            int height, int width) 
    {
        byte[] Y = new byte[height * width];
        for(int i=0; i<height; i++)
        for(int j=0; j<width; j++)
        {
            int index = i*width + j;
            float x1 = X1[index] & 0xff;
            float x2 = X2[j];
            Y[index] = (byte) (alpha*x1 + beta*x2 + gamma);
        }
        return Y;
    }
    
    public static byte[] img_linear2_field(byte[] X1, float[] X2,
            float alpha, float beta, float gamma,
            int height, int width) 
    {
        byte[] Y = new byte[height * width];
        for(int i=0; i<height; i++)
        for(int j=0; j<width; j++)
        {
            int index = i*width + j;
            float x1 = X1[index] & 0xff;
            float x2 = X2[i];
            Y[index] = (byte) (alpha*x1 + beta*x2 + gamma);
        }
        return Y;
    }
    
    public static float[] img_linear2_div_row(byte[] X, float[] X1,  float[] X2,
            float alpha1, float beta1, float gamma1,
            float alpha2, float beta2, float C,
            int height, int width) 
    {
        float[] Y = new float[height * width];
        for(int i=0; i<height; i++)
        for(int j=0; j<width; j++)
        {
            int index = i*width + j;
            float x1 = X1[j];
            float x2 = X2[j];
            float x = X[index] & 0xff;
            float y1 = (alpha1*x + beta1*x1 + gamma1);
            float y2 = (alpha2*x2 + beta2);
            Y[index] = y1 / y2 + C;
        }
        return Y;
    }
    
    public static float[] img_linear2_div_field(byte[] X, float[] X1,  float[] X2,
            float alpha1, float beta1, float gamma1,
            float alpha2, float beta2, float C,
            int height, int width) 
    {
        float[] Y = new float[height * width];
        for(int i=0; i<height; i++)
        for(int j=0; j<width; j++)
        {
            int index = i*width + j;
            float x1 = X1[i];
            float x2 = X2[i];
            float x = X[index] & 0xff;
            float y1 = (alpha1*x + beta1*x1 + gamma1);
            float y2 = (alpha2*x2 + beta2);
            Y[index] = y1 / y2 + C;
        }
        return Y;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Vector-Element-Operation">
    public static void elementAdd(double[] a, double[] b, double[] c)
    {for(int i=0;i<c.length;i++) c[i]=a[i]+b[i];}
    
    public static void elementAdd(float[] a, float[] b, float[] c)
    {for(int i=0;i<c.length;i++) c[i]=a[i]+b[i];}
    
    public static void assign(float[] a, float value) {
        for(int i=0; i<a.length; i++) a[i] = value;
    }
    
    public static void elementAdd(double alpha, double[] a, double beta ,double[] b, double[] c)
    {for(int i=0;i<c.length;i++) c[i]= alpha*a[i] + beta*b[i];}
    
    public static void elementAdd(float alpha, float[] a, float beta ,float[] b, float[] c)
    {for(int i=0;i<c.length;i++) c[i]= alpha*a[i] + beta*b[i];}
    
    
    public static void elementAddSquare(float[] a, float[] b, float[] c)
    {
        for(int i=0;i<c.length;i++) c[i] = a[i] + b[i]*b[i];
    }
     
    public static void elementAddSquare(float alpha, float[] a, float beta ,float[] b, float[] c)
    {
        for(int i=0;i<c.length;i++) c[i] = alpha*a[i] + beta*b[i]*b[i];
    }
    
     /**
     * <pre>
     * consider the input Array{@code left}, {@code right} as two vector
     * int the space with the same dimension, find the difference.
     * for each each components of
     * {@code left}, {@code right}:
     *      {@code result[i]=left[i]-right[i]}
     * </pre>
     * @param a
     * @param b 
     * @param c the difference between vector left and right
     */
    public static void elementSub(double[] a, double[] b, double[] c)
    {for(int i=0;i<c.length;i++) c[i]=a[i]-b[i];}
    
    public static void elementSub(float[] a, float[] b, float[] c)
    {for(int i=0;i<c.length;i++) c[i]=a[i]-b[i];}
    
    /**
     * c[i] = alpha*a[i] - beta*b[i]
     * @param alpha
     * @param a
     * @param beta
     * @param b
     * @param c 
     */
    public static void elementSub(double alpha, double[] a, double beta, double[] b, double[] c)
    {for(int i=0;i<c.length;i++) c[i] = alpha*a[i] - beta*b[i];}
    
    public static void elementSub(float alpha, float[] a, float beta, float[] b, float[] c)
    {for(int i=0;i<c.length;i++) c[i] = alpha*a[i] - beta*b[i];}
    
    /**
     * <pre>
     * This function may be widely used int Neural Network.
     * for each element of the input Arrays {@code left}, {@code right}:
     *      {@code c[i] = k * a[i] + (1-k) * b[i];}
     * </pre>
     * @param c
     * @param a
     * @param k
     * @param b 
     */
    public static void momentum(double[] c, double[] a, double k, double[] b)
    {Vector.elementAdd(k, a, 1-k, b, c);}
    
    /**
     * <pre>
     * Consider the two input Arrays {@code left}, {@code right} as two 
     * vectors, compute the Hadamard product of them.
     * for each component of {@code left}, {@code right}:
     *      {@code c[i] = a[i]*b[i]}
     * </pre>
     * @param a
     * @param b 
     * @param c
     */
    public static void elementMul(double[] a, double[] b, double[] c)
    {for(int i=0;i<c.length;i++) c[i]=a[i]*b[i];}
    
    public static void elementMul(float[] a, float[] b, float[] c)
    {for(int i=0;i<c.length;i++) c[i]=a[i]*b[i];}
    
    /**
     * c[i] = k*a[i]*b[i]
     * @param k
     * @param a
     * @param b
     * @param c 
     */
    public static void elementMul(double k, double[] a, double[] b, double[] c){
        for(int i=0;i<c.length;i++) c[i]=k*a[i]*b[i];
    }
    
    public static void elementMul(float k, float[] a, float[] b, float[] c){
        for(int i=0;i<c.length;i++) c[i]=k*a[i]*b[i];
    }
    
    public static void linear(float alpha, float[] a, float beta, float[] b){
        for(int i=0; i<a.length; i++) b[i] = alpha*a[i] + beta;
    }
    
    public static void elementMul(float a1, float[] A, float b1, 
            float a2, float[] B, float b2, float[] C)
    {
        for(int i=0;i<C.length;i++) C[i]=(a1*A[i] + b1)*(a2*B[i] + b2);
    }
    /**
     * <pre>
     * Consider the two input Arrays {@code left}, {@code right} as two 
     * vectors, compute the division by each component of them.
     * for each component of {@code left}, {@code right}:
     *      {@code c[i] = a[i]/b[i]}
     * </pre>
     * @param c
     * @param a
     * @param b 
     */
    public static void elementDiv(double[] a, double[] b, double[] c)
    {for(int i=0;i<c.length;i++) c[i]=a[i]/b[i];}
    
    public static void elementDiv(float[] a, float[] b, float[] c)
    {for(int i=0;i<c.length;i++) c[i]=a[i]/b[i];}
    
    /**
     * c[i] = k*a[i]/b[i].
     * @param k
     * @param c
     * @param a
     * @param b 
     */
    public static void elementDiv(float k, double[] a, double[] b, double[] c)
    {for(int i=0;i<c.length;i++) c[i] = k*a[i]/b[i];}
    
    public static void elementDiv(float k, float[] a, float[] b, float[] c)
    {for(int i=0;i<c.length;i++) c[i] = k*a[i]/b[i];}
    
    /**
     * b[i] = k/a[i].
     * @param k
     * @param a
     * @param b 
     */
    public static void elementRpl(float k, double[] a, double[] b)
    {for(int i=0;i<b.length;i++) b[i] = k/a[i];}
    
    public static void elementRpl(float k, float[] a, float[] b)
    {for(int i=0;i<b.length;i++) b[i] = k/a[i];}
    
    /**
     * b[i] = a[i]+k.
     * @param a
     * @param k
     * @param b 
     */
    public static void elementScalarAdd(double[] a, double k, double[] b)
    {for(int i=0;i<b.length;i++) b[i] = a[i]+k;}
    
    public static void elementScalarAdd(float[] a, float k, float[] b)
    {for(int i=0;i<b.length;i++) b[i] = a[i]+k;}
    
    /**
     * b[i] = alpha*a[i] + beta.
     * @param alpha
     * @param a
     * @param beta
     * @param b 
     */
    public static void elementScalarAdd(double alpha, double[] a, double beta, double[] b)
    {for(int i=0;i<b.length;i++) b[i] = alpha*a[i] + beta;}
    
    public static void elementScalarAdd(float alpha, float[] a, float beta, float[] b)
    {for(int i=0;i<b.length;i++) b[i] = alpha*a[i] + beta;}
    
    /**
     * b[i] = a[i]*k.
     * @param a
     * @param k
     * @param b 
     */
    public static void elementScalarMul(double[] a, double k, double[] b)
    {for(int i=0;i<b.length;i++) b[i] = a[i]*k;}
    
    public static void elementScalarMul(float[] a, float k, float[] b)
    {for(int i=0;i<b.length;i++) b[i] = a[i]*k;}
    
    /**
     * b[i] = a[i]/k.
     * @param a
     * @param k
     * @param b 
     */
    public static void elementScalarDiv(double[] a, double k, double[] b) {
        for(int i=0;i<b.length;i++) b[i] = a[i]/k;
    }
    
    public static void elementScalarDiv(float[] a, float k, float[] b) {
        for(int i=0;i<b.length;i++) b[i] = a[i]/k;
    }
    
    public static void Momentum(float[] W, float[] deltaW,
            float[] V, float a1, float a2,
            float lr_t, int length)
    {
        for(int i=0; i<length; i++) {
            V[i] = a1*V[i] + a2*deltaW[i];
            W[i] = W[i] - lr_t*V[i];
        }
    }
    
    public static void SGDMN(float[] W, float[] deltaW,
            float[] V, float momentum, float dampen, float nesterov, 
            float lr, int length)
    {
        float K = (nesterov * momentum) + (1.0f - nesterov);
        for(int i=0; i<length; i++) {
            V[i] = momentum*V[i] + (1 - dampen)*deltaW[i];
            float step = nesterov * deltaW[i] + K*V[i];
            W[i] -= lr*step;
        }
    }
    
    public static void RMSprop(float[] W, float[] deltaW,
            float[] S, float a1, float a2, float e,
            float k, int length)
    {
        for(int i=0;i<length;i++)
        {
            S[i] = a1*S[i] + a2*deltaW[i]*deltaW[i];
            W[i] = W[i] - k * deltaW[i]/((float)Math.sqrt(S[i]) + e);
        }
    }
    
    public static void Adam(float[] W, float[] deltaW,
            float[] V, float a1, float a2,
            float[] S, float b1, float b2, float eps,
            float lr, int length)
    {
        for(int i=0; i<length; i++) {
            float dw = deltaW[i];
            V[i] = a1*V[i] + a2*dw;
            S[i] = b1*S[i] + b2*dw*dw;
            W[i] = W[i] - lr * V[i] / ((float)Math.sqrt(S[i]) + eps);
        }
    }
    
    public static void Adamax(float[] W, float[] deltaW,
            float[] V, float a1, float a2,
            float[] S, float b1, float eps,
            float lr, int length)
    {
        for(int i=0; i<length; i++) {
            float dw = deltaW[i];
            V[i] = a1*V[i] + a2*dw;
            S[i] = Math.max(S[i] * b1, Math.abs(dw));
            W[i] = W[i] - lr * V[i] / (S[i] + eps);
        }
    }
     
    public static void Adamod(float[] W, float[] deltaW,
            float[] V, float a1, float a2,
            float[] S, float b1, float b2, float e,
            float[] G, float c1, float c2,
            float lr, int length)
    {
        for(int i=0;i<length;i++) {
            V[i] = a1*V[i] + a2*deltaW[i];
            S[i] = b1*S[i] + b2*deltaW[i]*deltaW[i];
            
            float neta = (float) (lr / (Math.sqrt(S[i]) + e));
            G[i] = c1*G[i] + c2*neta;
            
            W[i] -= Math.min(neta, G[i]) * V[i];
        }
    }
    
    
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Vector-Creator">
    //<editor-fold defaultstate="collapsed" desc="creator: arrayCopy">
    public static byte[] arrayCopy(byte[] a) {
        byte[] arr = new byte[a.length];
        System.arraycopy(a, 0, arr, 0, a.length);
        return arr;
    }
    public static byte[] arrayCopy(byte[] a, int low, int high) {
        byte[] arr = new byte[high - low + 1];
        System.arraycopy(a, low, arr, 0, high - low + 1);
        return arr;
    }
    
    public static char[] arrayCopy(char[] a) {
        char[] arr = new char[a.length];
        System.arraycopy(a, 0, arr, 0, a.length);
        return arr;
    }
    public static char[] arrayCopy(char[] a, int low, int high) {
        char[] arr = new char[high - low + 1];
        System.arraycopy(a, low, arr, 0, high - low + 1);
        return arr;
    }
    
    public static int[] arrayCopy(int[] a) {
        int[] arr = new int[a.length];
        System.arraycopy(a, 0, arr, 0, a.length);
        return arr;
    }
    public static int[] arrayCopy(int[] a, int low, int high) {
        int[] arr = new int[high - low + 1];
        System.arraycopy(a, low, arr, 0, high - low + 1);
        return arr;
    }

    public static long[] arrayCopy(long[] a) {
        long[] arr = new long[a.length];
        System.arraycopy(a, 0, arr, 0, a.length);
        return arr;
    }
    public static long[] arrayCopy(long[] a, int low, int high) {
        long[] arr = new long[high - low + 1];
        System.arraycopy(a, low, arr, 0, high - low + 1);
        return arr;
    }

    public static float[] arrayCopy(float[] a) {
        float[] arr = new float[a.length];
        System.arraycopy(a, 0, arr, 0, a.length);
        return arr;
    }
    public static float[] arrayCopy(float[] a, int low, int high) {
        float[] arr = new float[high - low + 1];
        System.arraycopy(a, low, arr, 0, high - low + 1);
        return arr;
    }

    public static double[] arrayCopy(double[] a) {
        double[] arr = new double[a.length];
        System.arraycopy(a, 0, arr, 0, a.length);
        return arr;
    }
    public static double[] arrayCopy(double[] a, int low, int high) {
        double[] arr = new double[high - low + 1];
        System.arraycopy(a, low, arr, 0, high - low + 1);
        return arr;
    }

    public static Object[] arrayCopy(Object[] a) {
        Object[] arr = new Object[a.length];
        System.arraycopy(a, 0, arr, 0, a.length);
        return arr;
    }
    public static Object[] arrayCopy(Object[] a, int low, int high) {
        Object[] arr = new Object[high - low + 1];
        System.arraycopy(a, low, arr, 0, high - low + 1);
        return arr;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="creator: extra">
    public static double[] sequence(int length, double base, double div) {
        double[] arr = new double[length]; arr[0] = base;
        for(int i=1; i<arr.length; i++) arr[i] = arr[i-1] + div;
        return arr;
    }
    
    public static int[] sequence(int length) { return sequence(length, 0, 1); }
    public static int[] sequence(int length, int base, int div) {
        int[] arr = new int[length]; arr[0] = base;
        for(int i=1; i<arr.length; i++) arr[i] = arr[i-1] + div;
        return arr;
    }
   
    public static <T> T[] sequence(int length, Class<T> clazz, SequenceCreator<T> sc) {
        T[] arr = (T[]) Array.newInstance(clazz, length);
        for(int i=0;i<arr.length;i++) arr[i] = sc.create(i);
        return arr;
    }
    
    public static float[] zeros(int length) { return new float[length]; }
    public static float[] ones(int length) { return constants(1.0f, length); }
    
    public static<T> Collection<T> collection(T[] arr) {
        ZArrayList r = new ZArrayList<>();
        for(T v:arr) r.add(v);
        return r;
    }
    
   
    
    public static float[] nan(int length) {
        float[] v = new float[length];
        for(int i=0; i<length; i++) v[i] = Float.NaN;
        return v;
    }
    
    public static float[] region(float[] X, int start, int end) {
        float[] Y = new float[end - start + 1];
        for(int i=start; i<=end; i++) Y[i] = X[i];
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="creator: append">
    public static int[] append(int[] arr, int v)  {
        if(arr == null) return new int[] { v };
        int[] arr2 = new int[arr.length + 1];
        System.arraycopy(arr, 0, arr2, 0, arr.length);
        arr2[arr.length] = v;
        return arr2;
    }
    public static int[] append(int v, int[] arr) {
        if(arr == null) return new int[] { v };
        int[] arr2 = new int[arr.length + 1];
        arr2[0] = v;
        System.arraycopy(arr, 0, arr2, 1, arr.length);
        return arr2;
    }
    
    public static long[] append(long[] arr, long v)  {
        if(arr == null) return new long[] { v };
        long[] arr2 = new long[arr.length + 1];
        System.arraycopy(arr, 0, arr2, 0, arr.length);
        arr2[arr.length] = v;
        return arr2;
    }
    public static long[] append(long v, long[] arr) {
        if(arr == null) return new long[] { v };
        long[] arr2 = new long[arr.length + 1];
        arr2[0] = v;
        System.arraycopy(arr, 0, arr2, 1, arr.length);
        return arr2;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="creator: constants">
    public static long[] constants(long C, int length) {
        long[] arr = new long[length];
        for(int i=0; i<arr.length; i++) arr[i] = C;
        return arr;
    }
    
    public static int[] constants(int C, int length) {
        int[] arr = new int[length];
        for(int i=0; i<arr.length; i++) arr[i] = C;
        return arr;
    }
    
    public static short[] constants(short C, int length) {
        short[] arr = new short[length];
        for(int i=0; i<arr.length; i++) arr[i] = C;
        return arr;
    }
    
    public static char[] constants(char C, int length) {
        char[] arr = new char[length];
        for(int i=0; i<arr.length; i++) arr[i] = C;
        return arr;
    }
    
    public static byte[] constants(byte C, int length) {
        byte[] arr = new byte[length];
        for(int i=0; i<arr.length; i++) arr[i] = C;
        return arr;
    }
    
    public static double[] constants(double C, int length) {
        double[] arr = new double[length];
        for(int i=0; i<arr.length; i++) arr[i] = C;
        return arr;
    }
    
    public static float[] constants(float C, int length) {
        float[] arr = new float[length];
        for(int i=0; i<arr.length; i++) arr[i] = C;
        return arr;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="creator: random">
    public static int[] random_int_vector(int width) { return exr().next_int_vector(width); }
    public static int[] random_int_vector(int width, int max) { return exr().next_int_vector(width, 0,  max); }
    public static int[] random_int_vector(int width, int min, int max ) { return exr().next_int_vector(width, min, max); }

    public static short[] random_short_vector(int length) {  return exr().next_short_vector(length); }
    public static short[] random_short_vector(int length, short max) { return exr().next_short_vector(length, max); }
    public static short[] random_short_vector(int length, short min, short max) { return exr().next_short_vector(length, min, max); }

    public static char[] random_char_vector(int length) {  return exr().next_char_vector(length); }
    public static char[] random_char_vector(int length, char max) { return exr().next_char_vector(length, max); }
    public static char[] random_char_vector(int length, char min, char max) { return exr().next_char_vector(length, min, max); }
    
    public static String random_string(int length) { return exr().nextString(length); }
    public static String random_string(int length, char min, char max) { return exr().nextString(length, min, max); }

    public static byte[] random_byte_vector(int length) {  return exr().next_byte_vector(length); }
    public static byte[] random_byte_vector(int length, byte max) { return exr().next_byte_vector(length, max); }
    public static byte[] random_byte_vector(int length, byte min, byte max) { return exr().next_byte_vector(length, min, max); }

    public static double[] random_double_vector(int length) { return exr().next_double_vector(length); }
    public static double[] random_double_vector(int length, double max) { return exr().next_double_vector(length, max); }
    public static double[] random_double_vector(int length, double min ,double max) { return exr().next_double_vector(length, min, max); }
    
    public static double[] random_gaussian_vector(int length) { return exr().next_gaussian_vector(length); }
    public static double[] random_gaussian_vector(int length, double max) { return exr().next_gaussian_vector(length, max); }
    public static double[] random_gaussian_vector(int length, double min, double max) { return exr().next_gaussian_vector(length, min, max); }

    public static float[] random_float_vector(int length) { return exr().next_float_vector(length); }
    public static float[] random_float_vector(int length, float max) { return exr().next_float_vector(length, 0, max); }
    public static float[] random_float_vector(int length, float min, float max) { return exr().next_float_vector(length, min, max); }
    
    public static float[] random_gaussianf_vector(int length) { return exr().next_gaussianf_vector(length); }
    public static float[] random_gaussianf_vector(int length, float max) { return exr().next_gaussianf_vector(length, max); }
    public static float[] random_gaussianf_vector(int length, float min, float max) { return exr().next_gaussianf_vector(length, min, max); }

    public static <T> T[] randomObjectVector(int length, Class<T> clazz, IntFunction<T> func){ return exr().next_object_vector(length, clazz, func); }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Vector-Manage-Function">
    //<editor-fold defaultstate="collapsed" desc="Next-Permutation">
    @Passed
    public static void reverse(int[] a, int low, int high)
    {
        for(int t;low<high;low++, high--)
            {t=a[low];a[low]=a[high];a[high]=t;}
    }
    public static void reverse(int[] a)
    {
        reverse(a, 0, a.length-1);
    }
    @Passed
    public static void reverse(char[] a, int low, int high)
    {
        for(char t;low<high;low++, high--)
            {t=a[low];a[low]=a[high];a[high]=t;}
    }
    public static void reverse(char[] a)
    {
        reverse(a, 0, a.length-1);
    }
    @Passed
    public static boolean nextPermutation(int[] a, int low, int high)
    {
        int cur=high, pre=cur-1;
        for(;cur>low && a[cur] <= a[pre]; cur--, pre--);
        if(cur<=low) return false;
        
        for(cur=high;a[cur]<=a[pre];cur--);
        int t=a[cur];a[cur]=a[pre];a[pre]=t;
        reverse(a, pre+1, high);//让最小的一段开头最大，结尾最小
        return true;
    }
    public static boolean nextPermutation(int[] a)
    {
        return nextPermutation(a, 0, a.length-1);
    }
    @Passed
    public static boolean nextPermutation(char[] a, int low, int high)
    {
        int cur=high, pre=cur-1;
        for(;cur>low&&a[cur]<=a[pre];cur--,pre--);
        if(cur<=low) return false;
        
        for(cur=high;a[cur]<=a[pre];cur--);
        char t=a[cur];a[cur]=a[pre];a[pre]=t;
        reverse(a, pre+1, high);
        return true;
    }
    public static boolean nextPermutation(char[] a)
    {
        return nextPermutation(a, 0, a.length-1);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Max-Multiple">
    public static final long DEF_MAX_MULTIPLE_MOD=1000000009;
    @Passed
    public static long maxMultiple(int[] a, int low, int high, int k, long mod)
    {
        if(a.length<k) throw new IAE("K is greater than the length of Array.");
        long r=1, sign=1;
        Sort.sort(a, low ,high);
        
        if((k&1)==1)
        {
            r=a[high--];k--;
            if(r<0) sign=-1;
        }
        for(long lows, highs;k>0;k-=2)
        {
            lows=a[low]*a[low+1];highs=a[high]*a[high-1];
            if(sign*lows>sign*highs) {r=(r*(lows%mod))%mod;low-=2;}
            else {r=(r*(highs%mod))%mod;high+=2;}
        }
        return r;
    }
    public static long maxMultiple(int[] a, int k, int mod)
    {
        return maxMultiple(a, 0, a.length-1, k, mod);
    }
    @Passed
    public static long maxMultiple(int[] a, int low, int high, int k)
    {
        if(a.length<k) throw new IAE("K is greater than the length of Array.");
        long r=1, sign=1;
        Sort.sort(a, low ,high);
        
        if((k&1)==1)
        {
            r=a[high--];k--;
            if(r<0) sign=-1;
        }
        for(long lows, highs;k>0;k-=2)
        {
            lows=a[low]*a[low+1];highs=a[high]*a[high-1];
            if(sign*lows>sign*highs) {r*=lows;low-=2;}
            else {r*=highs;high+=2;}
        }
        return r;
    }
    public static long maxMultiple(int[] a, int k)
    {
        return maxMultiple(a, 0, a.length-1, k);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Remove-Duplcate"> 
    public static int removeDuplcateIfSorted(int[] arr)
    {
        if(arr==null||arr.length==0) return 0;
        int last=arr[0];
        int j=1;
        for(int i=1;i<arr.length;i++)
            if(last!=arr[i]) last=arr[j++]=arr[i];
        return j;
    }
    public static int removeDuplcateIfSorted(double[] arr)
    {
        if(arr==null||arr.length==0) return 0;
        double last=arr[0];
        int j=1;
        for(int i=1;i<arr.length;i++)
            if(last!=arr[i]) last=arr[j++]=arr[i];
        return j;
    }
    public static int removeDuplcateIfSorted(Object[] arr)
    {
        if(arr==null||arr.length==0) return 0;
        Object last=arr[0];
        int j=1;
        for(int i=1;i<arr.length;i++)
            if(last!=arr[i]||!last.equals(arr[i])) last=arr[j++]=arr[i];
        return j;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="remove function">
    /**
     * <pre>
     * We used an optimized way to compact array, find all no-null
     * elements of the input Array {@code arr}, and move all them together
     * from the start of {@code arr}， without changing the order of elements
     * which is not null. 
     * This algorithm work like this:
     * (1)find the first element block in the list that's not null
     * (2)find the next null element block
     * (3)find move the block to take place of the null block
     * (4)looped
     * </pre>
     * @param arr
     * @param low the start index
     * @param high the end index
     * @return the length of new Array
     */
    @Passed
    public static int removeNull(Object[] arr, int low, int high)//checked
    {
        int start,end,nstart;
        //find the first element block in the list that's not null
        for(end=high;end>=low&&arr[end]==null;end--);
        //looped block move to take place of null block
        while(end>low)//if end==0, means there is no null element
        {
            //find the not null block
            for(start=end-1;start>=low&&arr[start]!=null;start--);
            if(start<low) break;//all element is not null

            //find the null block
            for(nstart=start-1;nstart>=low&&arr[nstart]==null;nstart--);
            
            //move block
            System.arraycopy(arr, start+1, arr, nstart+1, end-=start);
            end+=nstart;
        }
        System.out.println("old:"+end);
        return end;
    }
    /**
     * <pre>
     * The operating priciple is similar to {@link Vector#removeNull(java.lang.Object[], int, int) },
     * the only difference is: this function remove all elements which 
     * is equal to {@code value}.
     * </pre>
     * @param arr
     * @param val 
     * @param low the start index
     * @param high the end index
     * @return the length of new Array
     */
    @Passed
    public static int remove(Object[] arr, Object val, int low, int high)
    {
        if(val==null) return Vector.removeNull(arr, low, high);
        int start,end,nstart;
        //find the first element block in the list that isn't equal to value
        for(end=high;end>=low&&val.equals(arr[end]);end--);
        //looped block move to take place of equaled block
        while(end>low)//if end==0, means there is no null element
        {
            //find the not-equal block
            for(start=end-1;start>=low&&!val.equals(arr[start]);start--);
            if(start<low) break;//all element is not null

            //find the equaled block
            for(nstart=start-1;nstart>=low&&val.equals(arr[nstart]);nstart--);
            
            //move block
            System.arraycopy(arr, start+1, arr, nstart+1, end-=start);
            end+=nstart;
        }
        return end;
    }
    /**
     * <pre>
     * The operating priciple is similar to {@link Vector#removeNull(java.lang.Object[], int, int) },
     * the only difference is: this function remove all elements which 
     * is equal to {@code value}.
     * </pre>
     * @param arr
     * @param val 
     * @param low the start index
     * @param high the end index
     * @return the length of new Array
     */
    public static int remove(int[] arr, int val, int low, int high)
    {
        int start,end,nstart;
         //find the first element block in the list that isn't equal to value
        for(end=high;end>=low&&arr[end]==(val);end--);
        while(end>low)//if end==0, means there is no null element
        {
            //find the not-equal block
            for(start=end-1;start>=low&&arr[start]!=val;start--);
            if(start<low) break;//all element is not null

            //find the equaled block
            for(nstart=start-1;nstart>=low&&arr[nstart]==val;nstart--);
            
            //move block
            System.arraycopy(arr, start+1, arr, nstart+1, end-=start);
            end+=nstart;
        }
        return end;
    }
    /**
     * <pre>
     * The operating priciple is similar to {@link Vector#removeNull(java.lang.Object[], int, int) },
     * the only difference is: this function remove all elements which 
     * meets the need of {@code pre}.
     * </pre>
     * @param arr
     * @param pre
     * @param low the start index
     * @param high the end index
     * @return he length of new Array
     */
    @Passed
    public static int remove(Object[] arr, Predicate pre, int low, int high)
    {
        int start,end,nstart;
        //find the first element block in the list that doesn't meet pre
        for(end=high;end>=low&&pre.test(arr[end]);end--);
        //looped block move to take place of satisfied block
        while(end>low)
        {
            //find the not-satisfied block
            for(start=end-1;start>=low&&!pre.test(arr[start]);start--);
            if(start<low) break;
            
            //find the satisfied block
            for(nstart=start-1;nstart>low&&pre.test(arr[nstart]);nstart--);
            
            //move block
            System.arraycopy(arr, start+1, arr, nstart+1, end-=start);
            end+=nstart;
        }
        return high;
    }
      /**
     * <pre>
     * The operating priciple is similar to {@link Vector#removeNull(java.lang.Object[], int, int) },
     * the only difference is: this function remove all elements which 
     * meets the need of {@code pre} and {@code condition}:
     *      {@code pre.test(each element, condition)}.
     * </pre>
     * @param arr
     * @param pre
     * @param condition the second parameter of {@link BiPredicate#test(Object, Object) }
     * @param low the start index
     * @param high the end index
     * @return he length of new Array
     */
    @Passed
    public static int remove(Object[] arr, BiPredicate pre, Object condition, int low, int high)
    {
        int start,end,nstart;
        //find the first element block in the list that doesn't meet pre
        for(end=high;end>=low&&pre.test(arr[end], condition);end--);
        //looped block move to take place of satisfied block
        while(end>low)
        {
            //find the not-satisfied block
            for(start=end-1;start>=low&&!pre.test(arr[start], condition);start--);
            if(start<low) break;
            
            //find the satisfied block
            for(nstart=start-1;nstart>low&&pre.test(arr[nstart], condition);nstart--);
            
            //move block
            System.arraycopy(arr, start+1, arr, nstart+1, end-=start);
            end+=nstart;
        }
        return end;
    }
    
    public static int removeNull(Object[] arr, int high)
    {
        return Vector.removeNull(arr, 0, high);
    }
    public static int removeNull(Object[] arr)
    {
        return Vector.removeNull(arr, 0, arr.length-1);
    }
    
    public static int remove(Object[] arr, Object val, int high)
    {
        return Vector.remove(arr, val, 0, high);
    }
    public static int remove(Object[] arr, Object val)
    {
        return Vector.remove(arr, val, 0, arr.length-1);
    }
    
    public static int remove(int[] arr, int val, int high)
    {
        return Vector.remove(arr, val, 0, high);
    }
    public static int remove(int[] arr, int val)
    {
        return Vector.remove(arr, val, 0, arr.length-1);
    }
    
    public static int remove(Object[] arr, Predicate pre, int high)
    {
        return Vector.remove(arr, pre, 0, high);
    }
    public static int remove(Object[] arr, Predicate pre)
    {
        return Vector.remove(arr, pre, 0, arr.length-1);
    }
    
    public static int remove(Object[] arr, BiPredicate pre, Object condition, int high)
    {
        return Vector.remove(arr, pre, condition, 0, high);
    }
    public static int remove(Object[] arr, BiPredicate pre, Object condition)
    {
        return Vector.remove(arr, pre, condition, 0, arr.length-1);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Vector: check functions">
    //<editor-fold defaultstate="collapsed" desc="check: requireNonNull">
    public static void requireNonNull(Object[] arr, String name) { requireNonNull(arr, name, 0, arr.length - 1); }
    public static void requireNonNull(Object[] arr, String name, int low, int high) {
        if(arr == null) throw new NullPointerException(name + "is null");
        for(int i=low; i<= high; i++)
            if(arr[i] == null) throw new NullPointerException(name + "[" + i + "is null");
    }
    
    public static void requireNonNull(long[] arr, String name) {  requireNonNull(arr, name, 0, arr.length - 1); }
    public static void requireNonNull(long[] arr, String name, int low, int high) {
        if(arr == null) throw new NullPointerException(name + "is null");
        for(int i=low; i<= high; i++)
            if(arr[i] == 0L) throw new NullPointerException(name + "[" + i + "is null");
    }
    //</editor-fold>
   
    //<editor-fold defaultstate="collapsed" desc="check: tone_of_sequence">
    public static final int NOT_SURE = 0;
    public static final int INCREASE = 1;
    public static final int STRICT_INCREASE = 2;
    public static final int DECREASE = -1;
    public static final int STRICT_DECREASE = -2;
    
    public static int tone_of_sequence(int[] a) { return tone_of_sequence(a, 0 ,a.length - 1); }
    public static int tone_of_sequence(int[] a, int low, int high) {
        if(high <= low) return NOT_SURE;
        if(a[low + 1] > a[low]) {
            int base = STRICT_INCREASE;
            for(int i=low+1; i<high; i++) {
                if(a[i + 1] < a[i]) { base = NOT_SURE; break; }
                else if(a[i+1] == a[i]) base = INCREASE;
            }
            return base;
        }
        else {
            int base = STRICT_DECREASE;
            for(int i=low+1; i<high; i++) {
                if(a[i + 1] > a[i]) { base = NOT_SURE; break; }
                else if(a[i + 1] == a[i]) base = DECREASE;
            }
            return base;
        }
    }
    
    public static int toneOfSequence(double[] a) { return tone_of_sequence(a, 0 ,a.length - 1); }
    public static int tone_of_sequence(double[] a, int low, int high) {
        if(high <= low) return NOT_SURE;
        if(a[low + 1] >a [low])  {
            int base = STRICT_INCREASE;
            for(int i=low+1; i<high; i++) {
                if(a[i + 1] < a[i]) { base = NOT_SURE; break; }
                else if(a[i + 1] == a[i]) base = INCREASE;
            }
            return base;
        }
        else {
            int base = STRICT_DECREASE;
            for(int i=low+1; i<high; i++) {
                if(a[i + 1] > a[i]) { base = NOT_SURE; break; }
                else if(a[i + 1] == a[i]) base = DECREASE;
            }
            return base;
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="check: is_constant">
    public static boolean is_constant(float[] a, float v) {
        if(a == null || a.length == 0) return true;
        for(int i=0; i<a.length; i++) if(a[i] != v) return false;
        return true;
    }
    
    public static boolean is_constant(int[] a, int v) {
        if(a == null || a.length == 0) return true;
        for(int i=0; i<a.length; i++) if(a[i] != v) return false;
        return true;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="check: is_(strict)_desending">
    public static boolean is_desending(int... a) {return is_desending(a, 0, a.length-1);}
    public static boolean is_desending(int[] a, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        while(low < high) if(a[low] < a[++low]) return false;
        return true;
    }
    
    public static boolean is_desending(float... a) {return is_desending(a, 0, a.length - 1);}
    public static boolean is_desending(float[] a, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        while(low < high) if(a[low] < a[++low]) return false;
        return true;
    }
    
    public static boolean is_desending(double... a) {return is_desending(a, 0, a.length - 1);}
    public static boolean is_desending(double[] a, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        while(low < high) if(a[low] < a[++low]) return false;
        return true;
    }
    
    public static boolean is_desending(Comparable... a) {return is_desending(a, 0, a.length - 1);}
    public static boolean is_desending(Comparable[] a, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        while(low < high) if(a[low].compareTo(a[++low]) < 0) return false;
        return true;
    }
     
    public static boolean is_strict_desending(int... a) { return is_strict_desending(a, 0, a.length - 1); }
    public static boolean is_strict_desending(int[] a, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        while(low < high) if(a[low] <= a[++low]) return false;
        return true;
    }
    
    public static boolean is_strict_desending(float... a) { return is_strict_desending(a, 0, a.length - 1); }
    public static boolean is_strict_desending(float[] a, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        while(low < high) if(a[low] <= a[++low]) return false;
        return true;
    }
    
    public static boolean is_strict_desending(double... a) { return is_strict_desending(a, 0, a.length - 1); }
    public static boolean is_strict_desending(double[] a, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        while(low < high) if(a[low] <= a[++low]) return false;
        return true;
    }
     
    public static boolean is_strict_desending(Comparable... a) { return is_strict_desending(a, 0, a.length-1); }
    public static boolean is_strict_desending(Comparable[] a, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        while(low < high) if(a[low].compareTo(a[++low]) <= 0) return false;
        return true;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="check: is_(strict)_ascending">
    public static boolean is_ascending(int[] a) { return is_ascending(a, 0, a.length - 1); }
    public static boolean is_ascending(int[] a, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        while(low < high) if(a[low] > a[++low]) return false;
        return true;
    }
     
    public static boolean is_ascending(float[] a) { return is_ascending(a, 0, a.length - 1);}
    public static boolean is_ascending(float[] a, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        while(low < high) if(a[low] > a[++low]) return false;
        return true;
    }
    
    public static boolean is_ascending(double[] a) { return is_ascending(a, 0, a.length - 1);}
    public static boolean is_ascending(double[] a, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        while(low < high) if(a[low] > a[++low]) return false;
        return true;
    }
    
    public static boolean is_ascending(Comparable[] a) { return is_ascending(a, 0, a.length - 1); }
    public static boolean is_ascending(Comparable[] a, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        while(low < high) if(a[low].compareTo(a[++low]) > 0) return false;
        return true;
    }
    
    public static boolean is_strict_ascending(int[] a) { return is_strict_ascending(a, 0, a.length - 1); }
    public static boolean is_strict_ascending(int[] a, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        while(low < high) if(a[low] >= a[++low]) return false;
        return true;
    }
     
    public static boolean is_strict_ascending(float[] a) { return is_strict_ascending(a, 0, a.length - 1); }
    public static boolean is_strict_ascending(float[] a, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        while(low < high) if(a[low] >= a[++low]) return false;
        return true;
    }
    
    public static boolean is_strict_ascending(double[] a) { return is_strict_ascending(a, 0, a.length - 1); }
    public static boolean is_strict_ascending(double[] a, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        while(low < high) if(a[low] >= a[++low]) return false;
        return true;
    }
    
    public static boolean is_strict_ascending(Comparable[] a) { return is_strict_ascending(a, 0, a.length - 1); }
    public static boolean is_strict_ascending(Comparable[] a, int low, int high) {
        if(low > high) { int t = low; low = high; high = t; }
        while(low < high) if(a[low].compareTo(a[++low]) >= 0) return false;
        return true;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="check: samePercent">
    public static boolean PRINT_DIFFERENT = false;
    
    //<editor-fold defaultstate="collapsed" desc="samePercent_absolute">
    static final void print_dif_abosulte(int index, Object dif, Object a, Object b) {
        System.out.println("abosulte different ["  + index + "]: dif = " + dif + ",  a = " + a + ",  b = " + b);
    }
    
    public static float samePercent_absolute(byte[] A, byte[] B) { 
        int length = (A.length < B.length?  A.length : B.length);
        return samePercent_absolute(A, B, length, 0); 
    }
    public static float samePercent_absolute(byte[] A, byte[] B, int threshold) {
        int length = (A.length < B.length?  A.length : B.length);
        return samePercent_absolute(A, B, length, threshold); 
    }
    public static float samePercent_absolute(byte[] A, byte[] B, int length, int threshold) {
        if(length >= A.length) length = A.length;
        if(length >= B.length) length = B.length;
        
        int sum = 0;
        for(int i=0; i<length; i++) {
            if(A[i] == B[i]) { sum++; continue; }
            float dif = Math.abs(A[i] - B[i]);
            if(dif < threshold) sum++;
            else if(PRINT_DIFFERENT) print_dif_abosulte(i, dif, A[i], B[i]);
        }
        return ((float)sum) / length;
    }
    
    public static float samePercent_absolute(int[] A, int[] B) { 
        int length = (A.length < B.length?  A.length : B.length);
        return samePercent_absolute(A, B, length, 0); 
    }
    public static float samePercent_absolute(int[] A, int[] B, int threshold) { 
        int length = (A.length < B.length?  A.length : B.length);
        return samePercent_absolute(A, B, length, threshold); 
    }
    public static float samePercent_absolute(int[] A, int[] B, int length, int threshold) {
        if(length >= A.length) length = A.length;
        if(length >= B.length) length = B.length;
        
        int sum = 0;
        for(int i=0;i<length;i++) {
            if(A[i] == B[i]) { sum++; continue; }
            float dif = Math.abs(A[i] - B[i]);
            if(dif < threshold) sum++;
            else if(PRINT_DIFFERENT) print_dif_abosulte(i, dif, A[i], B[i]);
        }
        return ((float)sum) / length;
    }
   
    public static float samePercent_absolute(float[] A, float[] B) {
        int length = (A.length < B.length?  A.length : B.length);
        return samePercent_absolute(A, B, length, 1e-3f); 
    }
    public static float samePercent_absolute(float[] A, float[] B, float threshold) { 
        int length = (A.length < B.length?  A.length : B.length);
        return samePercent_absolute(A, B, length, threshold); 
    }
    public static float samePercent_absolute(float[] A, float[] B, int length, float threshold) {
        if(length >= A.length) length = A.length;
        if(length >= B.length) length = B.length;
        
        int sum=0;
        for(int i=0;i<length;i++) {
            if(A[i] == B[i]) { sum++; continue; }
            float dif = Math.abs(A[i] - B[i]);
            if(dif < threshold) sum++;
            else if(Float.isNaN(A[i]) && Float.isNaN(B[i])) sum++;
            else if(PRINT_DIFFERENT) print_dif_abosulte(i, dif, A[i], B[i]);
        }
        return ((float)sum)/length;
    }
    
    public static float difference_absolute(float[] A, float[] B) {
        int length = (A.length < B.length?  A.length : B.length);
        return difference_absolute(A, B, length);
    }
    public static float difference_absolute(float[] A, float[] B, int length) {
        if(length >= A.length) length = A.length;
        if(length >= B.length) length = B.length;
         
        double sum = 0;
        for(int i=0;i<length;i++) sum += Math.abs(A[i] - B[i]);
        return (float) (sum / length);
    }
    
    public static float difference_relative(float[] A, float[] B) {
        int length = (A.length < B.length?  A.length : B.length);
        return difference_relative(A, B, length);
    }
    public static float difference_relative(float[] A, float[] B, int length) {
        if(length >= A.length) length = A.length;
        if(length >= B.length) length = B.length;
         
        double sum = 0; 
        int count = 0;
        for(int i=0;i<length;i++) {
            float dif = Math.abs((A[i] - B[i]) / (B[i] + A[i]));
            if(dif != Float.NaN) { sum += dif; count++; }
        }
        return (float) (sum / count);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="samePercent_relative">
    static final void print_dif_relative(int index, Object dif, Object a, Object b) {
        System.out.println("relative different ["  + index + "]: dif = " + dif + ",  a = " + a + ",  b = " + b);
    }
    
    public static float samePercent_relative(float[] A, float[] B) { 
        int length = (A.length < B.length? A.length : B.length);
        return samePercent_relative(A, B, length, 1e-3f); 
    }
    public static float samePercent_relative(float[] A, float[] B, float threshold) {
        int length = (A.length < B.length? A.length : B.length);
        return samePercent_relative(A, B, length, threshold);
    }
    public static float samePercent_relative(float[] A, float[] B, int length, float threshold) {
        if(length < A.length) length = A.length;
        if(length < B.length) length = B.length;
        
        int sum = 0;
        for(int i=0;i<length;i++) {
            if(A[i] == B[i]) { sum++; continue; }
            float dif = Math.abs((A[i] - B[i] + Float.MIN_VALUE) / (A[i] + B[i] + Float.MIN_VALUE));
            if(dif < threshold) sum++;
            else if(Float.isNaN(A[i]) && Float.isNaN(B[i])) sum++;
            else if(PRINT_DIFFERENT) print_dif_relative(i, dif, A[i], B[i]);
        }
        return ((float)sum) / length;
    }
  
    public static float samePercent_relative(double[] A, double[] B) {
        int length = (A.length < B.length? A.length : B.length);
        return samePercent_relative(A, B, length, 1e-3f);
    }
    public static float samePercent_relative(double[] A, double[] B, double threshold) {
        int length = (A.length < B.length? A.length : B.length);
        return samePercent_relative(A, B, length, threshold);
    }
    public static float samePercent_relative(double[] A, double[] B, int length, double threshold) {
        if(length < A.length) length = A.length;
        if(length < B.length) length = B.length;
        
        int sum=0;
        for(int i=0; i<length; i++) {
            if(A[i] == B[i]) { sum++; continue; }
            double dif = Math.abs((A[i] - B[i] + Double.MIN_VALUE) / (A[i] + B[i] + Double.MIN_VALUE));
            if(dif < threshold) sum++;
            else if(Double.isNaN(A[i]) && Double.isNaN(B[i])) sum++;
            else if(PRINT_DIFFERENT) print_dif_relative(i, dif, A[i], B[i]);
        }
        return ((float)sum) / length;
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="check: zeroPercent">
    public static float zeroPercent(double... A) {return zeroPercent(A, 0, A.length - 1);}
    public static float zeroPercent(double[] A, int low, int high) {
        int sum = 0;
        for(int i=low; i<=high; i++) if(A[i] == 0) sum++;
        return ((float)sum) / (high - low + 1);
    }
    
    public static float zeroPercent(float... A) {return zeroPercent(A, 0, A.length - 1);}
    public static float zeroPercent(float[] A, int low, int high) {
        int sum = 0;
        for(int i=low; i<=high; i++) if(A[i] == 0) sum++;
        return ((float)sum) / (high - low + 1);
    }
    
    public static float zeroPercent(byte... A) {return zeroPercent(A, 0, A.length - 1);}
    public static float zeroPercent(byte[] A, int low, int high) {
        int sum = 0;
        for(int i=low; i<=high; i++) if(A[i] == 0) sum++;
        return ((float)sum) / (high - low + 1);
    }
    
    public static float zeroPercent(char... A) {return zeroPercent(A, 0, A.length - 1);}
    public static float zeroPercent(char[] A, int low, int high) {
        int sum = 0;
        for(int i=low; i<=high; i++) if(A[i] == 0) sum++;
        return ((float)sum) / (high - low + 1);
    }
    
    public static float zeroPercent(int... A) {return zeroPercent(A, 0, A.length - 1);}
    public static float zeroPercent(int[] A, int low, int high) {
        int sum = 0;
        for(int i=low; i<=high; i++) if(A[i] == 0) sum++;
        return ((float)sum) / (high - low + 1);
    }
    
    public static float zeroPercent(long... A) {return zeroPercent(A, 0, A.length - 1);}
    public static float zeroPercent(long[] A, int low, int high) {
        int sum = 0;
        for(int i=low; i<=high; i++) if(A[i] == 0) sum++;
        return ((float)sum) / (high - low + 1);
    }
    //</editor-fold>
    
    public static float matchPercent(float[] A, Predicate<Float> checker){ return matchPercent(A, 0, A.length - 1, checker); }
    public static float matchPercent(float[] A, int low, int high, Predicate<Float> checker) {
        int sum = 0;
        for(int i=low; i<=high; i++) if(checker.test(A[i])) sum++;
        return ((float)sum) / (high - low + 1);
    }
    
    public static int diversity(float[] A) {return diversity(A, 0, A.length - 1);}
    public static int diversity(float[] A, int low, int high) {
        HashSet<Float> set = new HashSet<>();
        for(int i=low; i<=high; i++) set.add(A[i]);
        return set.size();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Vector: sort">
    public static void sort(Comparable<?>... a) { Sort.sort(a, 0, a.length - 1); }
    public static void sort(Comparable<?>[] a, int low, int high) { Sort.sort(a, low, high); }
    
    public static <T> void sort(T[] a, Comparator<T> cmp) { Sort.sort(a, cmp, 0, a.length - 1); }
    public static <T> void sort(T[] a, Comparator<T> cmp, int low, int high) {  Sort.sort(a, cmp, low, high); }
    
    public static void sort(byte... a) { Sort.sort(a, 0, a.length - 1); }
    public static void sort(byte[] a, int low, int high) { Sort.sort(a, low, high); }
    
    public static void sort(char... a) { Sort.sort(a, 0, a.length - 1); }
    public static void sort(char[] a, int low, int high) { Sort.sort(a, low, high); }
    
    public static void sort(short... a) { Sort.sort(a, 0, a.length - 1); }
    public static void sort(short[] a, int low, int high) { Sort.sort(a, low, high); }
    
    public static void sort(int... a) { Sort.sort(a, 0, a.length - 1); }
    public static void sort(int[] a, int low, int high) { Sort.sort(a, low, high); }
    
    public static void sort(long... a) { Sort.sort(a, 0, a.length - 1); }
    public static void sort(long[] a, int low, int high) { Sort.sort(a, low, high); }
    
    public static void sort(float... a) { Sort.sort(a, 0, a.length - 1); }
    public static void sort(float[] a, int low, int high) { Sort.sort(a, low, high); }
    
    public static void sort(double... a) { Sort.sort(a, 0, a.length-1); }
    public static void sort(double[] a, int low, int high) { Sort.sort(a, low, high); }
    
    public static <T> void sort(char[] key, T[] val) { Sort.sort(key, val, 0, key.length - 1); }
    public static <T> void sort(char[] key, T[] val, int low, int high) { Sort.sort(key, val, low, high); } 
    
    public static <T> void sort(int[] key, T[] val) { Sort.sort(key, val, 0, key.length - 1); }
    public static <T> void sort(int[] key, T[] val, int low, int high) { Sort.sort(key, val, low, high); } 
    
    public static <T> void sort(float[] key, T[] val) { Sort.sort(key, val, 0, key.length - 1); }
    public static <T> void sort(float[] key, T[] val, int low, int high) { Sort.sort(key, val, low, high); } 
    
    public static <T> void sort(double[] key, T[] val) { Sort.sort(key, val, 0, key.length - 1); }
    public static <T> void sort(double[] key, T[] val, int low, int high) { Sort.sort(key, val, low, high); } 
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="tensor operation">
    //<editor-fold defaultstate="collapsed" desc="flattenL float">
    public static float[] flatten(float[][] mat) {
        int dim0 = mat.length, dim1 = mat[0].length;
        float[] v = new float[dim0 * dim1];
        int index = 0;
        for(int i=0; i<dim0; i++)
            for(int j=0; j<dim1; j++)
                v[index++] = mat[i][j];
        return v;
    }
    
    public static float[] flatten(float[][][] tense) {
        int dim0 = tense.length;
        int dim1 = tense[0].length;
        int dim2 = tense[0][0].length;
        float[] v = new float[dim0 * dim1 * dim2];
        int index = 0;
        for(int i=0; i<dim0; i++)
            for(int j=0; j<dim1; j++)
                for(int k=0; k<dim2; k++)
                    v[index++] = tense[i][j][k];
        return v;
    }
    
    public static float[] flatten(float[][][][] tense) {
        int dim0 = tense.length;
        int dim1 = tense[0].length;
        int dim2 = tense[0][0].length;
        int dim3 = tense[0][0][0].length;
        
        float[] v = new float[dim0 * dim1 * dim2 * dim3];
        int index = 0;
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    for(int d3=0; d3<dim3; d3++)
                    v[index++] = tense[d0][d1][d2][d3];
        return v;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="flatten: double">
    public static double[] flatten(double[][][] tense) {
        int dim0 = tense.length;
        int dim1 = tense[0].length;
        int dim2 = tense[0][0].length;
        double[] v = new double[dim0 * dim1 * dim2];
        int index = 0;
        for(int i=0; i<dim0; i++)
            for(int j=0; j<dim1; j++)
                for(int k=0; k<dim2; k++)
                    v[index++] = tense[i][j][k];
        return v;
    }
    
     public static double[] flatten(double[][][][] tense) {
        int dim0 = tense.length;
        int dim1 = tense[0].length;
        int dim2 = tense[0][0].length;
        int dim3 = tense[0][0][0].length;
        
        double[] v = new double[dim0 * dim1 * dim2 * dim3];
        int index = 0;
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    for(int d3=0; d3<dim3; d3++)
                    v[index++] = tense[d0][d1][d2][d3];
        return v;
    }
     //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="flatten: byte">
    public static byte[] flatten(byte[][] mat) {
        int dim0 = mat.length, dim1 = mat[0].length;
        byte[] v = new byte[dim0 * dim1];
        int index = 0;
        for(int i=0; i<dim0; i++)
            for(int j=0; j<dim1; j++)
                v[index++] = mat[i][j];
        return v;
    }
    
    public static byte[] flatten(byte[][][] tense) {
        int dim0 = tense.length;
        int dim1 = tense[0].length;
        int dim2 = tense[0][0].length;
        byte[] v = new byte[dim0 * dim1 * dim2];
        int index = 0;
        for(int i=0; i<dim0; i++)
            for(int j=0; j<dim1; j++)
                for(int k=0; k<dim2; k++)
                    v[index++] = tense[i][j][k];
        return v;
    }
    
    public static byte[] flatten(byte[][][][] tense) {
        int dim0 = tense.length;
        int dim1 = tense[0].length;
        int dim2 = tense[0][0].length;
        int dim3 = tense[0][0][0].length;
        
        byte[] v = new byte[dim0 * dim1 * dim2 * dim3];
        int index = 0;
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    for(int d3=0; d3<dim3; d3++)
                    v[index++] = tense[d0][d1][d2][d3];
        return v;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="toND(byte)">  
    public static byte[][] to2D(byte[] X, int dim0, int dim1) {
        byte[][] Y = new byte[dim0][dim1];
        int index = 0;
        for(int i=0; i<dim0; i++)
            for(int j=0; j<dim1; j++)
                Y[i][j] = X[index++];//X[index++]
        return Y;
    }
    
    public static byte[][][] to3D(byte[] X, int dim0, int dim1, int dim2) {
        byte[][][] Y = new byte[dim0][dim1][dim2];
        int index = 0;
        for(int i=0; i<dim0; i++)
            for(int j=0; j<dim1; j++)
                for(int k=0; k<dim2; k++)
                    Y[i][j][k] = X[index++];
        return Y;
    }
    
    public static byte[][][][] to4D(byte[] X, int dim0, int dim1, int dim2, int dim3) {
        byte[][][][] Y = new byte[dim0][dim1][dim2][dim3];
        int index = 0;
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    for(int d3=0; d3<dim3; d3++)
                        Y[d0][d1][d2][d3] = X[index++];
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="toND(int)">  
    public static int[][] to2D(int[] X, int dim0, int dim1) {
        int[][] Y = new int[dim0][dim1];
        int index = 0;
        for(int i=0; i<dim0; i++)
            for(int j=0; j<dim1; j++)
                    Y[i][j] = X[index++];//X[index++]
        return Y;
    }
    
    public static int[][][] to3D(int[] X, int dim0, int dim1, int dim2) {
        int[][][] Y = new int[dim0][dim1][dim2];
        int index = 0;
        for(int i=0; i<dim0; i++)
            for(int j=0; j<dim1; j++)
                for(int k=0; k<dim2; k++)
                    Y[i][j][k] = X[index++];
        return Y;
    }
    
    public static int[][][][] to4D(int[] X, int dim0, int dim1, int dim2, int dim3) {
        int[][][][] Y = new int[dim0][dim1][dim2][dim3];
        int index = 0;
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    for(int d3=0; d3<dim3; d3++)
                        Y[d0][d1][d2][d3] = X[index++];
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="toND(float)">  
    public static float[][] to2D(float[] X, int dim0, int dim1) {
        float[][] Y = new float[dim0][dim1]; int index = 0;
        if (dim1 < 64) {
            for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                Y[d0][d1] = X[index++];
        }
        else {
            for(int d0=0; d0<dim0; d0++) {
                System.arraycopy(X, index, Y[d0], 0, dim1);
                index += dim1;
            }
        }
        return Y;
    }
    
    public static float[][][] to3D(float[] X, int dim0, int dim1, int dim2) {
        float[][][] Y = new float[dim0][dim1][dim2]; int index = 0;
        if (dim2 < 64) {
            for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
            for(int d2=0; d2<dim2; d2++)
                Y[d0][d1][d2] = X[index++];
        }
        else {
            for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++) {
                System.arraycopy(X, index, Y[d0][d1], 0, dim2);
                index += dim2;
            }
        }
        return Y;
    }
    
    public static float[][][][] to4D(float[] X, int dim0, int dim1, int dim2, int dim3) {
        float[][][][] Y = new float[dim0][dim1][dim2][dim3]; int index = 0;
        if (dim3 < 64) {
            for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
            for(int d2=0; d2<dim2; d2++)
            for(int d3=0; d3<dim3; d3++)
                Y[d0][d1][d2][d3] = X[index++];
        }
        else {
            for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
            for(int d2=0; d2<dim2; d2++) {
                System.arraycopy(X, index, Y[d0][d1][d2], 0, dim3);
                index += dim3;
            }
        }
        return Y;
    }
    //</editor-fold>
    
    public static double[][][][] to4D_double(float[] X, int dim0, int dim1, int dim2, int dim3) {
        double[][][][] tense = new double[dim0][dim1][dim2][dim3];
        int index = 0;
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    for(int d3=0; d3<dim3; d3++)
                        tense[d0][d1][d2][d3] = X[index++];
        return tense;
    }
    
    
    public static float[][][] reshape3D(float[][][][] tense, int dim0, int dim1, int dim2) {
        float[] v = Vector.flatten(tense);
        return Vector.to3D(v, dim0, dim1, dim2);
    }
    
    //<editor-fold defaultstate="collapsed" desc="Tensor transpose<float>">
    public static float[][][][] transpose4D(float[][][][] X, int dimIdx1, int dimIdx2) {
        if(dimIdx1 > dimIdx2) {
            int t = dimIdx1; dimIdx1 = dimIdx2; dimIdx2 = t;
        }
        
        int dim0 = X.length;
        int dim1 = X[0].length;
        int dim2 = X[0][0].length;
        int dim3 = X[0][0][0].length;
        
        if(dimIdx1 == 0 && dimIdx2 == 1) return transepose4D_0_1(X, dim0, dim1, dim2, dim3);
        if(dimIdx1 == 0 && dimIdx2 == 2) return transepose4D_0_2(X, dim0, dim1, dim2, dim3);
        if(dimIdx1 == 0 && dimIdx2 == 3) return transepose4D_0_3(X, dim0, dim1, dim2, dim3);
        if(dimIdx1 == 1 && dimIdx2 == 2) return transepose4D_1_2(X, dim0, dim1, dim2, dim3);
        if(dimIdx1 == 1 && dimIdx2 == 3) return transepose4D_1_3(X, dim0, dim1, dim2, dim3);
        if(dimIdx1 == 2 && dimIdx2 == 3) return transepose4D_2_3(X, dim0, dim1, dim2, dim3);
        else throw new IllegalArgumentException();
    }
    
    //<editor-fold defaultstate="collapsed" desc="transpose 4D">
    static float[][][][] transepose4D_0_1(float[][][][] X, int dim0, int dim1, int dim2, int dim3)
    {
        float[][][][] Y = new float[dim1][dim0][dim2][dim3];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    System.arraycopy(X[d0][d1][d2], 0, Y[d1][d0][d2], 0, dim3);
        return Y;
    }
    static float[][][][] transepose4D_0_2(float[][][][] X, int dim0, int dim1, int dim2, int dim3)
    {
        float[][][][] Y = new float[dim2][dim1][dim0][dim3];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    System.arraycopy(X[d0][d1][d2], 0, Y[d2][d1][d0], 0, dim3);
        return Y;
    }
    static float[][][][] transepose4D_0_3(float[][][][] X, int dim0, int dim1, int dim2, int dim3)
    {
        float[][][][] Y = new float[dim3][dim1][dim2][dim0];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    for(int d3=0; d3<dim3; d3++)
                        Y[d3][d1][d2][d0] = X[d0][d1][d2][d3];
        return Y;
    }
    
    static float[][][][] transepose4D_1_2(float[][][][] X, int dim0, int dim1, int dim2, int dim3)
    {
        float[][][][] Y = new float[dim0][dim2][dim1][dim3];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    System.arraycopy(X[d0][d1][d2], 0, Y[d0][d2][d1], 0, dim3);
        return Y;
    }
    
    static float[][][][] transepose4D_1_3(float[][][][] X, int dim0, int dim1, int dim2, int dim3)
    {
        float[][][][] Y = new float[dim0][dim3][dim2][dim1];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    for(int d3=0; d3<dim3; d3++)
                        Y[d0][d3][d2][d1] = X[d0][d1][d2][d3];
        return Y;
    }
    
    static float[][][][] transepose4D_2_3(float[][][][] X, int dim0, int dim1, int dim2, int dim3)
    {
        float[][][][] Y = new float[dim0][dim1][dim3][dim2];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    for(int d3=0; d3<dim3; d3++)
                        Y[d0][d1][d3][d2] = X[d0][d1][d2][d3];
        return Y;
    }
    //</editor-fold>
    
    public static float[][][] transpose3D(float[][][] X, int dimIdx1, int dimIdx2) {
        if(dimIdx1 > dimIdx2) {
            int t = dimIdx1; dimIdx1 = dimIdx2; dimIdx2 = t;
        }
        
        int dim0 = X.length;
        int dim1 = X[0].length;
        int dim2 = X[0][0].length;
        
        if(dimIdx1 == 0 && dimIdx2 == 1) return transepose3D_0_1(X, dim0, dim1, dim2);
        if(dimIdx1 == 0 && dimIdx2 == 2) return transepose3D_0_2(X, dim0, dim1, dim2);
        if(dimIdx1 == 1 && dimIdx2 == 2) return transepose3D_1_2(X, dim0, dim1, dim2);
        else throw new IllegalArgumentException();
    }
    //<editor-fold defaultstate="collapsed" desc="transpose 3D">
    static float[][][] transepose3D_0_1(float[][][] X, int dim0, int dim1, int dim2)
    {
        float[][][] Y = new float[dim1][dim0][dim2];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                        Y[d1][d0][d2] = X[d0][d1][d2];
        return Y;
    }
    
    
    static float[][][] transepose3D_0_2(float[][][] X, int dim0, int dim1, int dim2)
    {
        float[][][] Y = new float[dim2][dim1][dim0];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                        Y[d2][d1][d0] = X[d0][d1][d2];
        return Y;
    }
    
    static float[][][] transepose3D_1_2(float[][][] X, int dim0, int dim1, int dim2)
    {
        float[][][] Y = new float[dim0][dim2][dim1];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                        Y[d0][d2][d1] = X[d0][d1][d2];
        return Y;
    }
    //</editor-fold>
    
    public static float[][] transpose2D(float[][] X) {
        int dim0 = X.length;
        int dim1 = X[0].length;
        
        float[][] Y = new float[dim1][dim0];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++) Y[d1][d0] = X[d0][d1];
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Image transpose<byte>">
    public static byte[][][][] img_transpose4D(byte[][][][] X, int dimIdx1, int dimIdx2) {
        if(dimIdx1 > dimIdx2) {
            int t = dimIdx1; dimIdx1 = dimIdx2; dimIdx2 = t;
        }
        
        int dim0 = X.length;
        int dim1 = X[0].length;
        int dim2 = X[0][0].length;
        int dim3 = X[0][0][0].length;
        
        if(dimIdx1 == 0 && dimIdx2 == 1) return img_transepose4D_0_1(X, dim0, dim1, dim2, dim3);
        if(dimIdx1 == 0 && dimIdx2 == 2) return img_transepose4D_0_2(X, dim0, dim1, dim2, dim3);
        if(dimIdx1 == 0 && dimIdx2 == 3) return img_transepose4D_0_3(X, dim0, dim1, dim2, dim3);
        if(dimIdx1 == 1 && dimIdx2 == 2) return img_transepose4D_1_2(X, dim0, dim1, dim2, dim3);
        if(dimIdx1 == 1 && dimIdx2 == 3) return img_transepose4D_1_3(X, dim0, dim1, dim2, dim3);
        if(dimIdx1 == 2 && dimIdx2 == 3) return img_transepose4D_2_3(X, dim0, dim1, dim2, dim3);
        else throw new IllegalArgumentException();
    }
    
    //<editor-fold defaultstate="collapsed" desc="img_transpose 4D">
    static byte[][][][] img_transepose4D_0_1(byte[][][][] X, int dim0, int dim1, int dim2, int dim3) {
        byte[][][][] Y = new byte[dim1][dim0][dim2][dim3];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    System.arraycopy(X[d0][d1][d2], 0, Y[d1][d0][d2], 0, dim3);
        return Y;
    }
    
    static byte[][][][] img_transepose4D_0_2(byte[][][][] X, int dim0, int dim1, int dim2, int dim3) {
        byte[][][][] Y = new byte[dim2][dim1][dim0][dim3];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    System.arraycopy(X[d0][d1][d2], 0, Y[d2][d1][d0], 0, dim3);
        return Y;
    }
    
    static byte[][][][] img_transepose4D_0_3(byte[][][][] X, int dim0, int dim1, int dim2, int dim3) {
        byte[][][][] Y = new byte[dim3][dim1][dim2][dim0];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    for(int d3=0; d3<dim3; d3++)
                        Y[d3][d1][d2][d0] = X[d0][d1][d2][d3];
        return Y;
    }
    
    static byte[][][][] img_transepose4D_1_2(byte[][][][] X, int dim0, int dim1, int dim2, int dim3) {
        byte[][][][] Y = new byte[dim0][dim2][dim1][dim3];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    System.arraycopy(X[d0][d1][d2], 0, Y[d0][d2][d1], 0, dim3);
        return Y;
    }
    
    static byte[][][][] img_transepose4D_1_3(byte[][][][] X, int dim0, int dim1, int dim2, int dim3) {
        byte[][][][] Y = new byte[dim0][dim3][dim2][dim1];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    for(int d3=0; d3<dim3; d3++)
                        Y[d0][d3][d2][d1] = X[d0][d1][d2][d3];
        return Y;
    }
    
    static byte[][][][] img_transepose4D_2_3(byte[][][][] X, int dim0, int dim1, int dim2, int dim3) {
        byte[][][][] Y = new byte[dim0][dim1][dim3][dim2];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    for(int d3=0; d3<dim3; d3++)
                        Y[d0][d1][d3][d2] = X[d0][d1][d2][d3];
        return Y;
    }
    //</editor-fold>
    
    public static byte[][][] img_transpose3D(byte[][][] X, int dimIdx1, int dimIdx2) {
        if(dimIdx1 > dimIdx2) { int t = dimIdx1; dimIdx1 = dimIdx2; dimIdx2 = t; }
        
        int dim0 = X.length;
        int dim1 = X[0].length;
        int dim2 = X[0][0].length;
        
        if(dimIdx1 == 0 && dimIdx2 == 1) return img_transepose3D_0_1(X, dim0, dim1, dim2);
        if(dimIdx1 == 0 && dimIdx2 == 2) return img_transepose3D_0_2(X, dim0, dim1, dim2);
        if(dimIdx1 == 1 && dimIdx2 == 2) return img_transepose3D_1_2(X, dim0, dim1, dim2);
        else throw new IllegalArgumentException();
    }
    //<editor-fold defaultstate="collapsed" desc="transpose 3D">
    static byte[][][] img_transepose3D_0_1(byte[][][] X, int dim0, int dim1, int dim2) {
        byte[][][] Y = new byte[dim1][dim0][dim2];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                        Y[d1][d0][d2] = X[d0][d1][d2];
        return Y;
    }
    
    static byte[][][] img_transepose3D_0_2(byte[][][] X, int dim0, int dim1, int dim2) {
        byte[][][] Y = new byte[dim2][dim1][dim0];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                        Y[d2][d1][d0] = X[d0][d1][d2];
        return Y;
    }
    
    static byte[][][] img_transepose3D_1_2(byte[][][] X, int dim0, int dim1, int dim2) {
        byte[][][] Y = new byte[dim0][dim2][dim1];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                        Y[d0][d2][d1] = X[d0][d1][d2];
        return Y;
    }
    //</editor-fold>
    
    public static byte[][] transpose2D(byte[][] X) {
        int dim0 = X.length;
        int dim1 = X[0].length;
        
        byte[][] Y = new byte[dim1][dim0];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++) Y[d1][d0] = X[d0][d1];
        return Y;
    }
    //</editor-fold>
    
    public static float[][][][] concat4D(int dimIdx, float[][][][]...X)  {
        if(dimIdx < 0) dimIdx += 4;
        if(dimIdx == 0) return concat4D_0(X);
        if(dimIdx == 1) return concat4D_1(X);
        if(dimIdx == 2) return concat4D_2(X);
        if(dimIdx == 3) return concat4D_3(X);
        throw new IllegalArgumentException();
    }
    
    //<editor-fold defaultstate="collapsed" desc="concat 4D">
    public static float[][][][] concat4D_0(float[][][][]... X) {
        int dimSize = 0;
        for(int i=0; i<X.length;i++) dimSize += X[i].length;
        
        int dim1 = X[0][0].length;
        int dim2 = X[0][0][0].length;
        int dim3 = X[0][0][0][0].length;
        
        float[][][][] Y = new float[dimSize][dim1][dim2][dim3];
        int yd0 = 0;
        for(int i=0; i<X.length; i++) {
            int dim0 = X[i].length;
            for(int d0=0; d0<dim0; d0++, yd0++)
            for(int d1=0; d1<dim1; d1++)
            for(int d2=0; d2<dim2; d2++) 
                System.arraycopy(X[i][d0][d1][d2], 0, Y[yd0][d1][d2], 0, dim3);
        }
        return Y;
    }
    
    public static float[][][][] concat4D_1(float[][][][]... X) {
        int dimSize = 0;
        for(int i=0; i<X.length; i++) dimSize += X[i][0].length;
        
        int dim0 = X[0].length;
        int dim2 = X[0][0][0].length;
        int dim3 = X[0][0][0][0].length;
        
        float[][][][] Y = new float[dim0][dimSize][dim2][dim3];
        int yd1 = 0;
        for(int i=0; i<X.length; i++){
            int dim1 = X[i][0].length;
            for(int d1=0; d1<dim1; d1++, yd1++)
            for(int d0=0; d0<dim0; d0++)
            for(int d2=0; d2<dim2; d2++)
                System.arraycopy(X[i][d0][d1][d2], 0, Y[d0][yd1][d2], 0, dim3);
        }
        return Y;
    }
    
    public static float[][][][] concat4D_2(float[][][][]... X) {
        int dimSize = 0;
        for(int i=0; i<X.length; i++) dimSize += X[i][0][0].length;
        
        int dim0 = X[0].length;
        int dim1 = X[0][0].length;
        int dim3 = X[0][0][0][0].length;
        
        float[][][][] Y = new float[dim0][dim1][dimSize][dim3];
        int yd2 = 0;
        for(int i=0; i<X.length; i++) {
            int dim2 = X[i][0][0].length;
            for(int d2=0; d2<dim2; d2++, yd2++)
            for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                System.arraycopy(X[i][d0][d1][d2], 0, Y[d0][d1][yd2], 0, dim3);
        }
        return Y;
    }
    
    public static float[][][][] concat4D_3(float[][][][]...X){
        int dimSize = 0;
        for(int i=0; i<X.length; i++) dimSize += X[i][0][0][0].length; 
        
        int dim0 = X[0].length;
        int dim1 = X[0][0].length;
        int dim2 = X[0][0][0].length;
        
        float[][][][] Y = new float[dim0][dim1][dim2][dimSize];
        int yd3 = 0;
        for(int i=0; i<X.length; i++) {
            int dim3 = X[i][0][0][0].length;
            for(int d3=0; d3<dim3; d3++, yd3++)
            for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
            for(int d2=0; d2<dim2; d2++)
                Y[d0][d1][d2][yd3] = X[i][d0][d1][d2][d3];
        }
        return Y;
    }
    //</editor-fold>
    
    public static float[][][][][] split4D(float[][][][] X, int dimIdx, int[] section) {
        if(dimIdx < 0) dimIdx += 4;
        if(dimIdx == 0) return split4D_0(X, section);
        if(dimIdx == 1) return split4D_1(X, section);
        if(dimIdx == 2) return split4D_2(X, section);
        if(dimIdx == 3) return split4D_3(X, section);
        throw new IllegalArgumentException();
    }
    
    //<editor-fold defaultstate="collapsed" desc="split 4d">
    public static float[][][][][] split4D_0(float[][][][] X, int[] section) 
    {
        int dim1 = X[0].length;
        int dim2 = X[0][0].length;
        int dim3 = X[0][0][0].length;
        
        float[][][][][] Y = new float[section.length][][][][];
        for(int i=0; i<section.length; i++) {
            Y[i] = new float[section[i]][dim1][dim2][dim3];
        }
        
        int yd0 = 0;
        for(int i=0; i<section.length; i++)
        {
            int dim0 = section[i];
            for(int d0=0; d0<dim0; d0++, yd0++)
            for(int d1=0; d1<dim1; d1++)
            for(int d2=0; d2<dim2; d2++)
                System.arraycopy(X[yd0][d1][d2], 0, Y[i][d0][d1][d2], 0, dim3);
                
        }
        return Y;
    }
    
    public static float[][][][][] split4D_1(float[][][][] X, int[] section) 
    {
        int dim0 = X.length;
        int dim2 = X[0][0].length;
        int dim3 = X[0][0][0].length;
        
        float[][][][][] Y = new float[section.length][][][][];
        for(int i=0; i<section.length; i++) {
            Y[i] = new float[dim0][section[i]][dim2][dim3];
        }
        
        int yd1 = 0;
        for(int i=0; i<section.length; i++)
        {
            int dim1 = section[i];
            for(int d1=0; d1<dim1; d1++, yd1++)
            for(int d0=0; d0<dim0; d0++)
            for(int d2=0; d2<dim2; d2++)
                System.arraycopy(X[d0][yd1][d2], 0, Y[i][d0][d1][d2], 0, dim3);
                
        }
        return Y;
    }
    
    public static float[][][][][] split4D_2(float[][][][] X, int[] section) 
    {
        int dim0 = X.length;
        int dim1 = X[0].length;
        int dim3 = X[0][0][0].length;
        
        float[][][][][] Y = new float[section.length][][][][];
        for(int i=0; i<section.length; i++) {
            Y[i] = new float[dim0][dim1][section[i]][dim3];
        }
        
        int yd2 = 0;
        for(int i=0; i<section.length; i++)
        {
            int dim2 = section[i];
            for(int d2=0; d2<dim2; d2++, yd2++)
            for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                System.arraycopy(X[d0][d1][yd2], 0, Y[i][d0][d1][d2], 0, dim3);
        }
        return Y;
    }
     
    public static float[][][][][] split4D_3(float[][][][] X, int[] section) 
    {
        int dim0 = X.length;
        int dim1 = X[0].length;
        int dim2 = X[0][0].length;
        
        float[][][][][] Y = new float[section.length][][][][];
        for(int i=0; i<section.length; i++) {
            Y[i] = new float[dim0][dim1][dim2][section[i]];
        }
        
        int yd3 = 0;
        for(int i=0; i<section.length; i++)
        {
            int dim3 = section[i];
            for(int d3=0; d3<dim3; d3++, yd3++)
            for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
            for(int d2=0; d2<dim2; d2++)
                Y[i][d0][d1][d2][d3] = X[d0][d1][d2][yd3];
        }
        return Y;
    }
    //</editor-fold>
    
    public static float[][][][] rot180(float[][][][] X) {
        int dim0 = X.length;
        int dim1 = X[0].length;
        int dim2 = X[0][0].length;
        int dim3 = X[0][0][0].length;
        
        float[][][][] Y = new float[dim0][dim1][dim2][dim3];
        for(int d0 = 0; d0 < dim0; d0++)
        for(int d1 = 0; d1 < dim1; d1++)
        for(int d2 = 0; d2 < dim2; d2++)
        for(int d3 = 0; d3 < dim3; d3++)
            Y[d0][d1][d2][d3] = X[d0][dim1 - 1 - d1][dim2 - 1 - d2][d3];
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Vector: expand functions">
    public static float[] expand_from_head(float[] a, int n) {//pad 0s at head
        if(a == null) return new float[n];
        if(a.length >= n) return a;
        float[] np = new float[n];
        for(int i=a.length-1, j=n-1; i>=0; i--, j--) np[j] = a[i];
        return np;
    }
    
   public static int[] expand_from_head(int[] a, int n) {//pad 0s at head
        if(a == null) return new int[n];
        if(a.length >= n) return a;
        int[] np = new int[n];
        for(int i=a.length-1, j=n-1; i>=0; i--, j--) np[j] = a[i];
        return np;
    }
    
    public static int[] expand_from_head(int[] a, int[] b) {//[b] [a], pad b.elems at head of a
        if(a == null || a.length == 0) return Vector.arrayCopy(b);
        if(a.length >= b.length) return a;
        
        int n = b.length, m = b.length - a.length; 
        int[] np = new int[n];
        for(int i=0; i<m; i++) np[i] = b[i];
        for(int i=m, j=0; i<n; i++, j++) np[i] = a[j];
        return np;
    }
    
    public static int[] expand_from_head_positive(int[] a, int[] b) {//[b] [a], pad b.elems at head of a
        if(a == null || a.length == 0) return Vector.arrayCopy(b);
        
        int n = b.length, m = b.length - a.length; 
        int[] np = new int[n];
        for(int i=0; i<m; i++) np[i] = b[i];
        for(int i=m, j=0; i<n; i++, j++)  np[i] = (a[j] > 0 ? a[j] : b[i]);
        return np;
    }
    //</editor-fold>
}