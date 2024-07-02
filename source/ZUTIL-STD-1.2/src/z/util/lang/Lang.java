/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.lang;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintStream;
import java.io.Serializable;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.net.JarURLConnection;
import java.net.URL;
import java.net.URLDecoder;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Collection;
import java.util.Date;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import z.util.factory.Meta;
import z.util.function.Converter;
import z.util.function.Printer;
import z.util.function.Stringer;
import z.util.math.ExRandom;
import z.util.math.vector.Matrix;
import z.util.math.vector.Vector;
import z.util.lang.annotation.Passed;

/**
 * <pre>
 * z.util.lang.Lang is the base of all class in Package z.util, before using
 * relative class, you must load z.util.lang.Lang.
 * (1)The convert-function and class-mapping are necessary to init
 * (2)while the print-function and to-String function is optional to load, according
 * to the configuration of 'z.util.lang.lang-site.xml'
 * </pre>
 * @author dell
 */
@SuppressWarnings("unchecked")
public final class Lang 
{
    public static final String NULL = "null";
    public static final String NULL_LN = "null\n";
    
    public static final int INT_BYTE = Integer.SIZE/Byte.SIZE;
    public static final int LENGTH_TIMES_INT_CHAR = Integer.SIZE / Character.SIZE;
    public static final int LENGTH_DEF_INT_CHAR = Integer.SIZE - Character.SIZE;
    
    public static final int KB_BIT  = 1  << 13;
    public static final int MB_BIT  = 1  << 23;
    public static final long GB_BIT = 1L << 33;
    
    private Lang() {}
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
    private static int rand_index = -1;
    private static final ExRandom[] randoms = new ExRandom[16];
    static { for(int i=0; i<16; i++) randoms[i] = new ExRandom(); }
    
    public static final ExRandom exr() { 
        rand_index = (rand_index + 1) & 15;
        return randoms[rand_index]; 
    }
    
    public static final boolean test(Object v) {
        if      (v instanceof Boolean)    return (Boolean) v;
        else if (v instanceof Number)     return ((Number)v).intValue() != 0; 
        else if (v instanceof Collection) return ((Collection)v).isEmpty();
        else if (v instanceof Map)        return((Map)v).isEmpty();
        else return v != null;
    }
    
    public static final void line() { DEF_OUT.println("------------------------------------------"); }
    
    public static final void line(char c) {
        char[] cs = new char[10];
        for(int i=0; i<cs.length; i++) cs[i]=c;
        DEF_OUT.println(new String(cs));
    }
    
    private static final SimpleDateFormat DEF_SDF = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    public static String currentDateTime() { return DEF_SDF.format(new Date()); }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="HashMap with Function-Pointer">
    //<editor-fold defaultstate="collapsed" desc="class Converter">
    //elementary converters-----------------------------------------------------
    private static final Converter doubleConverter = new Converter() {
        @Override
        public <T> T convert(String val) {
            return val == null ? null : (T) Double.valueOf(val);
        }
    };
    private static final Converter intConverter = new Converter() {
        @Override
        public <T> T convert(String val) {
            return val == null ? null : (T) Integer.valueOf(val);
        }
    };
    private static final Converter booleanConverter = new Converter() {
        @Override
        public <T> T convert(String val) {
            return val == null ? null : (T) Boolean.valueOf(val);
        }
    };
    private static final Converter floatConverter = new Converter() {
        @Override
        public <T> T convert(String val) {
            return val == null ? null : (T) Float.valueOf(val);
        }
    };
    private static final Converter longConverter = new Converter() {
        @Override
        public <T> T convert(String val) {
            return val == null ? null : (T) Long.valueOf(val);
        }
    };
    private static final Converter byteConverter = new Converter() {
        @Override
        public <T> T convert(String val) {
            return val == null ? null : (T) Byte.valueOf(val);
        }
    };
    private static final Converter shortConverter = new Converter() {
        @Override
        public <T> T convert(String val) {
            return val == null ? null : (T) Short.valueOf(val);
        }
    };
    private static final Converter stringConverter = new Converter() {
        @Override
        public <T> T convert(String val) {
            return (T) val;
        }
    };
    private static final Converter clazzConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return (T) Class.forName(val);
        }
    };

    //vector converters---------------------------------------------------------
    private static final Converter doubleVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Vector.to_double_vector(val);
        }
    };
    private static final Converter intVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Vector.to_int_vector(val);
        }
    };
    private static final Converter booleanVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Vector.to_boolean_vector(val);
        }
    };
    private static final Converter floatVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Vector.to_float_vector(val);
        }
    };
    private static final Converter longVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Vector.to_long_vector(val);
        }
    };
    private static final Converter byteVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Vector.to_byte_vector(val);
        }
    };
    private static final Converter shortVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Vector.to_short_vector(val);
        }
    };
    private static final Converter stringVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return (val == null ? null : (T) val.split(" {0,}, {0,}"));
        }
    };

    private static final Converter nDoubleVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Vector.to_Double_vector(val);
        }
    };
    private static final Converter nIntVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Vector.to_Integer_vector(val);
        }
    };
    private static final Converter nBooleanVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Vector.to_Boolean_vector(val);
        }
    };
    private static final Converter nFloatVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Vector.to_Float_vector(val);
        }
    };
    private static final Converter nLongVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Vector.to_Long_vector(val);
        }
    };
    private static final Converter nByteVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Vector.to_Byte_vector(val);
        }
    };
    private static final Converter nShortVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Vector.to_Short_vector(val);
        }
    };

    //matrix converters---------------------------------------------------------
    private static final Converter doubleMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Matrix.valueOfDoubleMatrix(val);
        }
    };
    private static final Converter intMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Matrix.valueOfIntMatrix(val);
        }
    };
    private static final Converter booleanMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Matrix.valueOfBooleanMatrix(val);
        }
    };
    private static final Converter floatMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Matrix.valueOfFloatMatrix(val);
        }
    };
    private static final Converter longMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Matrix.valueOfLongMatrix(val);
        }
    };
    private static final Converter byteMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Matrix.valueOfByteMatrix(val);
        }
    };
    private static final Converter shortMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Matrix.valueOfShortMatrix(val);
        }
    };
    private static final Converter stringMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return (val == null ? null : (T) Matrix.valueOfStringMatrix(val));
        }
    };

    private static final Converter nDoubleMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Matrix.valueOfNDoubleMatrix(val);
        }
    };
    private static final Converter nIntMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Matrix.valueOfNIntMatrix(val);
        }
    };
    private static final Converter nBooleanMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Matrix.valueOfNBooleanMatrix(val);
        }
    };
    private static final Converter nFloatMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Matrix.valueOfNFloatMatrix(val);
        }
    };
    private static final Converter nLongMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Matrix.valueOfNLongMatrix(val);
        }
    };
    private static final Converter nByteMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Matrix.valueOfNByteMatrix(val);
        }
    };
    private static final Converter nShortMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception {
            return val == null ? null : (T) Matrix.valueOfNShortMatrix(val);
        }
    };
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class Stringer">
    private static final Stringer booleanVectorStringer = (Stringer<boolean[]>) (val) -> Vector.toString(val);
    private static final Stringer byteVectorStringer = (Stringer<byte[]>) (val) -> Vector.toString(val);
    private static final Stringer shortVectorStringer = (Stringer<short[]>) (val) -> Vector.toString(val);
    private static final Stringer intVectorStringer = (Stringer<int[]>) (val) -> Vector.toString(val);
    private static final Stringer longVectorStringer = (Stringer<long[]>) (val) -> Vector.toString(val);
    private static final Stringer floatVectorStringer = (Stringer<float[]>) (val) -> Vector.toString(val);
    private static final Stringer doubleVectorStringer = (Stringer<double[]>) (val) -> Vector.toString(val);
    
    private static final Stringer booleanMatrixStringer = (Stringer<boolean[][]>) (boolean[][] val) -> Matrix.toString(val);
    private static final Stringer byteMatrixStringer = (Stringer<byte[][]>) (byte[][] val) -> Matrix.toString(val);
    private static final Stringer shortMatrixStringer = (Stringer<short[][]>) (short[][] val) -> Matrix.toString(val);
    private static final Stringer intMatrixStringer = (Stringer<int[][]>) (int[][] val) -> Matrix.toString(val);
    private static final Stringer longMatrixStringer = (Stringer<long[][]>) (long[][] val) -> Matrix.toString(val);
    private static final Stringer floatMatrixStringer = (Stringer<float[][]>) (float[][] val) -> Matrix.toString(val);
    private static final Stringer doubleMatrixStringer = (Stringer<double[][]>) (double[][] val) -> Matrix.toString(val);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class Printer"> 
    private static final Printer booleanVectorPrinter = (Printer<boolean[]>) (out, val) -> { Vector.println(out, val); };
    private static final Printer byteVectorPrinter = (Printer<byte[]>) (out, val) -> { Vector.println(out, val); };
    private static final Printer shortVectorPrinter = (Printer<short[]>) (out, val) -> { Vector.println(out, val); };
    private static final Printer intVectorPrinter = (Printer<int[]>) (out, val) -> { Vector.println(out, val); };
    private static final Printer longVectorPrinter = (Printer<long[]>) (out, val) -> { Vector.println(out, val); };
    private static final Printer floatVectorPrinter = (Printer<float[]>) (out, val) -> { Vector.println(out, val); };
    private static final Printer doubleVectorPrinter = (Printer<double[]>) (out, val) -> { Vector.println(out, val); };
    
    private static final Printer booleanMatrixPrinter = (Printer<boolean[][]>) (out, val) -> { Matrix.println(out, val); };
    private static final Printer byteMatrixPrinter = (Printer<byte[][]>) (out, val) -> { Matrix.println(out, val); };
    private static final Printer shortMatrixPrinter = (Printer<short[][]>) (out, val) -> { Matrix.println(out, val); };
    private static final Printer intMatrixPrinter = (Printer<int[][]>) (out, val) -> { Matrix.println(out, val); };
    private static final Printer longMatrixPrinter = (Printer<long[][]>) (out, val) -> { Matrix.println(out, val); };
    private static final Printer floatMatrixPrinter = (Printer<float[][]>) (out, val) -> { Matrix.println(out, val); };
    private static final Printer doubleMatrixPrinter = (Printer<double[][]>) (out, val) -> { Matrix.println(out, val); };
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Init-State">
    private static final String TOSTRING_INIT_CONF = "lang.toString.init";
    private static final String PRINT_INIT_CONF = "lang.print.init";
    
    static {
        try {
            Lang.init_ClassMapping();
            Lang.init_Converter();
            Meta mt = Meta.valueOf("z/util/lang/conf/zlang-site.xml", null, "configuration");
            if (mt.getValue(TOSTRING_INIT_CONF)) Lang.init_toString();
            if (mt.getValue(PRINT_INIT_CONF)) Lang.init_Print();
        }
        catch(Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Fail to init z.util.lang.Lang");
        }
    }
    //</editor-fold>
    
    private static Set<Class> ELEMENT_TYPE;
    private static Set<Class> ELEMENT_VECTOR_TYPE;
    private static Set<Class> ELEMENT_MATRIX_TYPE;
    private static Map<Class, String> CLASS_NAME_MAP;
    private static Map<String, Class> NAME_CLASS_MAP;
    private synchronized static void init_ClassMapping() {
        ELEMENT_TYPE = new HashSet<>();
        ELEMENT_VECTOR_TYPE = new HashSet<>();
        ELEMENT_MATRIX_TYPE = new HashSet<>();
        CLASS_NAME_MAP = new HashMap<>();
        NAME_CLASS_MAP = new HashMap<>();
        
        //for element data type-------------------------------------------------
        Class[] cls = Lang.elem_class();
        String[] name = Lang.elem_class_name();
        for(int i=0; i<cls.length; i++) {
            ELEMENT_TYPE.add(cls[i]);
            CLASS_NAME_MAP.put(cls[i], name[i]);
            NAME_CLASS_MAP.put(name[i], cls[i]);
        }
        
        //for element vector-----------------------------------------------------
        cls = Lang.elem_array_class();
        name = Lang.elem_array_class_name();
        for(int i=0; i<cls.length; i++)  {
            ELEMENT_VECTOR_TYPE.add(cls[i]);
            CLASS_NAME_MAP.put(cls[i], name[i]);
            NAME_CLASS_MAP.put(name[i], cls[i]);
        }
        
        //for element matrix----------------------------------------------------
        cls = Lang.elem_mat_class();
        name = Lang.elem_mat_class_name();
        for(int i=0; i<cls.length; i++)  {
            ELEMENT_MATRIX_TYPE.add(cls[i]);
            CLASS_NAME_MAP.put(cls[i], name[i]);
            NAME_CLASS_MAP.put(name[i], cls[i]);
        }
    }

    private static HashMap<Class,Converter> CLASS_CONVERTER_MAP;
    private static HashMap<String,Converter> NAME_CONVERTER_MAP;  
    private synchronized static void init_Converter() {
        CLASS_CONVERTER_MAP = new HashMap<>();
        NAME_CONVERTER_MAP = new HashMap<>();
        
        //for elemnt converter--------------------------------------------------
        Converter[] converter = new Converter[]{
            byteConverter, booleanConverter, shortConverter, intConverter, longConverter,
            floatConverter, doubleConverter,
            stringConverter, clazzConverter,
            booleanConverter, byteConverter, shortConverter, intConverter, longConverter,
            floatConverter, doubleConverter
        };
         
        Class[] cls = Lang.elem_class();
        String[] name = Lang.elem_class_name();
        for(int i=0; i<cls.length; i++)  {
            CLASS_CONVERTER_MAP.put(cls[i], converter[i]);
            NAME_CONVERTER_MAP.put(name[i], converter[i]);
        }
        
        //for vector converter--------------------------------------------------
        converter = new Converter[]{
            byteVectorConverter, booleanVectorConverter, shortVectorConverter, intVectorConverter, longVectorConverter,
            floatVectorConverter, doubleVectorConverter,
            stringVectorConverter,
            nBooleanVectorConverter, nByteVectorConverter, nShortVectorConverter, nIntVectorConverter, nLongVectorConverter,
            nFloatVectorConverter, nDoubleVectorConverter
        };
        
        cls = Lang.elem_array_class();
        name = Lang.elem_array_class_name();
        for(int i=0; i<cls.length; i++)   {
            CLASS_CONVERTER_MAP.put(cls[i], converter[i]);
            NAME_CONVERTER_MAP.put(name[i], converter[i]);
        }
        
        //for matrix converter--------------------------------------------------
        converter=new Converter[] {
            byteMatrixConverter, booleanMatrixConverter, shortMatrixConverter, intMatrixConverter, longMatrixConverter,
            floatMatrixConverter, doubleMatrixConverter,
            stringMatrixConverter,
            nBooleanMatrixConverter, nByteMatrixConverter, nShortMatrixConverter, nIntMatrixConverter, nLongMatrixConverter,
            nFloatMatrixConverter, nDoubleMatrixConverter
        };
        
        cls = Lang.elem_mat_class();
        name = Lang.elem_mat_class_name();
        for(int i=0; i<cls.length; i++)  {
            CLASS_CONVERTER_MAP.put(cls[i], converter[i]);
            NAME_CONVERTER_MAP.put(name[i], converter[i]);
        }
    }

    private static boolean TOSTRING_INIT = false;
    private static HashMap<Class, Stringer> CLASS_STRINGER_MAP;
    public synchronized static void init_toString()  {
        if(TOSTRING_INIT)  { System.out.println("z.util.Lang-Stringer has been initialized, don't call this function repeatedly"); return; }
        CLASS_STRINGER_MAP = new HashMap<>();
        
        //for vector Stringer---------------------------------------------------
        Stringer[] stringer = new Stringer[]{
            booleanVectorStringer, byteVectorStringer, shortVectorStringer, intVectorStringer, longVectorStringer,
            floatVectorStringer, doubleVectorStringer
        };
        
        Class[] cls = Lang.elem_array_class();
        for(int i=0; i<stringer.length; i++) CLASS_STRINGER_MAP.put(cls[i], stringer[i]);

        //for matrix Stringer---------------------------------------------------
        stringer = new Stringer[]{
            booleanMatrixStringer, byteMatrixStringer, shortMatrixStringer, intMatrixStringer, longMatrixStringer,
            floatMatrixStringer, doubleMatrixStringer
        };

        cls = Lang.elem_mat_class();
        for(int i=0; i<stringer.length; i++) CLASS_STRINGER_MAP.put(cls[i], stringer[i]);
        TOSTRING_INIT = true;
    }
    public synchronized static void cleanup_toString() {
        if(!TOSTRING_INIT)  {
            System.out.println("z.util.Lang-Stringer has been cleaned up,"
                    + " don't call this function repeatedly");
            return;
        }
        CLASS_STRINGER_MAP.clear();
        CLASS_STRINGER_MAP = null;
        TOSTRING_INIT = false;
    }
    
    private static boolean PRINT_INIT = false;
    private static HashMap<Class, Printer> CLASS_PRINTER_MAP;
    public synchronized static void init_Print() {
        if(PRINT_INIT) { System.out.println("z.util.Lang-Printer has been initialized, don't call this function repeatedly"); return; }
        CLASS_PRINTER_MAP = new HashMap<>();
        
        //for vector Stringer---------------------------------------------------
        Printer[] printer = new Printer[]{
            booleanVectorPrinter, byteVectorPrinter, shortVectorPrinter, intVectorPrinter, longVectorPrinter,
            floatVectorPrinter, doubleVectorPrinter
        };
        Class[] cls = Lang.elem_array_class();
        for (int i=0; i<printer.length; i++) CLASS_PRINTER_MAP.put(cls[i], printer[i]);

        //for matrix Stringer---------------------------------------------------
        printer = new Printer[]{
            booleanMatrixPrinter, byteMatrixPrinter, shortMatrixPrinter, intMatrixPrinter, longMatrixPrinter,
            floatMatrixPrinter, doubleMatrixPrinter};
        cls = Lang.elem_mat_class();
        for (int i=0; i<printer.length; i++) CLASS_PRINTER_MAP.put(cls[i], printer[i]);
        
        PRINT_INIT = true;
    }
    public synchronized static void cleanup_print() {
        if(!PRINT_INIT)  { System.out.println("z.util.Lang-Printer has been initialized, don't call this function repeatedly"); return; }
        CLASS_PRINTER_MAP.clear();
        CLASS_PRINTER_MAP = null;
        PRINT_INIT = false;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Reflect-Function">
    //<editor-fold defaultstate="collapsed" desc="Normal-Function">
    private static final Class<?>[] elem_cls = {
        byte.class, boolean.class, short.class, int.class, long.class,
        float.class, double.class,
        String.class, Class.class,
        Byte.class, Boolean.class, Short.class, Integer.class, Long.class,
        Float.class, Double.class 
    };
    private static final String[] elem_cls_name = {
        "byte", "boolean", "short", "int", "long",
        "float", "double",
        "String", "Class",
        "Byte", "Boolean", "Short", "Integer", "Long",
        "Float", "Double" 
    };
    
    private static final Class<?>[] elem_array_cls = {
        byte[].class, boolean[].class, short[].class, int[].class, long[].class,
        float[].class, double[].class,
        String[].class, 
        Byte[].class, Boolean[].class, Short[].class, Integer[].class, Long[].class,
        Float[].class, Double[].class 
    };
    private static final String[] elem_array_cls_name = {
        "byte[]", "boolean[]", "short[]", "int[]","long[]",
        "float[]", "double[]",
        "String[]",
        "Byte[]", "Boolean[]", "Short[]", "Integer[]", "Long[]",
        "Float[]", "Double[]" 
    };
    
    private static final Class<?>[] elem_mat_cls = {
        byte[][].class, boolean[][].class, short[][].class, int[][].class, long[][].class,
        float[][].class, double[][].class,
        String[][].class,
        Byte[][].class, Boolean[][].class, Short[][].class, Integer[][].class, Long[][].class,
        Float[][].class, Double[][].class 
    };
    private static final String[] elem_mat_cls_name = {
        "byte[][]", "boolean[][]", "short[][]", "int[][]","long[][]",
        "float[][]", "double[][]",
        "String[][]",
        "Byte[][]", "Boolean[][]", "Short[][]", "Integer[][]", "Long[][]",
        "Float[][]", "Double[][]" 
    };
    
    public static Class<?>[] elem_class() { return elem_cls; }
    public static Class<?>[] elem_array_class() { return elem_array_cls;}
    public static Class<?>[] elem_mat_class() { return elem_mat_cls; }

    public static String[] elem_class_name() { return elem_cls_name; }
    public static String[] elem_array_class_name() { return elem_array_cls_name; }
    public static String[] elem_mat_class_name() { return elem_mat_cls_name; }
    
    public static String class_name(Object o) {
        Class<?> cls = o.getClass();
        String name = CLASS_NAME_MAP.get(cls);
        return (name != null ? name : cls.getSimpleName());
    }
    
    public static boolean is_elem_type(Class<?> cls) { return ELEMENT_TYPE.contains(cls); }
    public static boolean is_elem_type(Field field) { return ELEMENT_TYPE.contains(field.getType()); }
    public static boolean is_elem_array_type(Class<?> cls) { return ELEMENT_VECTOR_TYPE.contains(cls); }
    public static boolean is_elem_array_type(Field field) { return ELEMENT_VECTOR_TYPE.contains(field.getType()); }
    public static boolean is_elem_mat_type(Class<?> cls) { return ELEMENT_MATRIX_TYPE.contains(cls); }
    public static boolean is_elem_mat_type(Field field) { return ELEMENT_MATRIX_TYPE.contains(field.getType()); }
    
    public static Class<?> is_array(Class<?> cls) {
        if(Lang.is_elem_array_type(cls)) return cls.getComponentType();
        if(!cls.isArray()) return null;
        else return cls.getComponentType();
    }
   
    public static Class<?> is_matrix(Class<?> cls) {
        if(Lang.is_elem_mat_type(cls)) return cls.getComponentType().getComponentType();
        Class<?> d1 = Lang.is_array(cls);
        return (d1 == null ? null : Lang.is_array(d1));
    }
    
    public static boolean is_sub_cls(Class<?> sub_cls, Class<?> super_cls) {
        try { sub_cls.asSubclass(super_cls); return true; }
        catch(Exception e) { return false; }
    }
    public static String to_detail_String(Class<?> cls)  {
        StringBuilder sb = new StringBuilder(1024);
        sb.append("Class - ").append(cls.getName()).append("{\n");
        sb.append("Fields:\n");
        Vector.appendLn(sb, getExtensiveFields(cls));
        
        sb.append("\nFunctions\n");
        Vector.append(sb, cls.getDeclaredMethods());
        sb.append(" }");
        return sb.toString();
    }

    public static String to_generic_String(Class<?> cls)  {
        StringBuilder sb = new StringBuilder(1024);
        sb.append("Class - ").append(cls.getName()).append("{\n");
        
        Collection<Field> fields = Lang.getExtensiveFields(cls);
        sb.append("Fields:\n");
        Class<?>[] interfs = null;
        for(Field field : fields) {
            sb.append(field.toGenericString());
            interfs = field.getType().getInterfaces();
            if(interfs != null) sb.append(Arrays.toString(interfs));
            sb.append('\n');
        }
        
        Method[] methods = cls.getDeclaredMethods();
        sb.append("\nFunctions\n");
        for (Method m : methods) sb.append(m.toGenericString()).append('\n');
        sb.append(" }");
        return sb.toString();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Field-Function">
    public static final Predicate fieldIsStatic = (Predicate<Field>) (Field t) -> Modifier.isStatic(t.getModifiers());
    public static final Predicate fieldNotStatic = (Predicate<Field>) (Field t) -> !Modifier.isStatic(t.getModifiers());
    public static final Predicate fieldIsFinal = (Predicate<Field>) (Field t) -> Modifier.isFinal(t.getModifiers());
    public static final Predicate fieldNotFinal = (Predicate<Field>) (Field t) -> !Modifier.isFinal(t.getModifiers());
    
    //<editor-fold defaultstate="collapsed" desc="Extensive:Core-Code">
    /**
     * get all declared fields of a specified class, both its or its 
     * super classes'.
     * @param cls 
     * @return 
     */
    public static Collection<Field> getExtensiveFields(Class cls)
    {
        Objects.requireNonNull(cls);
        LinkedList<Field> c = new LinkedList<>();
        
        for(; cls !=null && cls != Object.class; cls = cls.getSuperclass()) {
            for(Field fid : cls.getDeclaredFields()) c.add(fid);
        }
        return c;
    }
     /**
     * get all declared fields of a specified class, both its or its 
     * super classes'. for each field you can use {@code Predicate pre} to check
     * and decide whether to add it to the Collection.
     * @param cls 
     * @param pre 
     * @return 
     */
    public static Collection<Field> getExtensiceFields(Class cls, Predicate<Field> pre)
    {
        Objects.requireNonNull(cls);
        Collection<Field> c=new LinkedList<>();
        
        Field[] fids=null;
        for(int i; cls!=null && cls!=Object.class; cls=cls.getSuperclass())
        {
            fids = cls.getDeclaredFields();
            for(i=0;i<fids.length;i++)
                if(pre.test(fids[i])) c.add(fids[i]);
        }
        return c;
    }
    /**
     * get all declared fields of a specified class, both its or its 
     * super classes'. for each field you can use {@code Predicate pre} to check
     * and decide whether to add it to the Collection. Besides, before you add
     * the field to the Collection, you can use con to process the field, like
     * {@code field.setAccessible(true);} and so on.
     * @param cls 
     * @param pre 
     * @param con 
     * @return 
     */
    public static Collection<Field> getExtensiceFields(Class cls, Predicate<Field> pre, Consumer<Field> con)
    {
        Objects.requireNonNull(cls);
        Collection<Field> c=new LinkedList<>();
        
        Field[] fids=null;
        for(int i;cls!=null&&cls!=Object.class;cls=cls.getSuperclass())
        {
            fids=cls.getDeclaredFields();
            for(i=0;i<fids.length;i++)
                if(pre.test(fids[i])) {con.accept(fids[i]);c.add(fids[i]);}
        }
        return c;
    }
    //</editor-fold>
    public static Collection<Field> extensive_member_fields(Class cls) { return Lang.getExtensiceFields(cls, fieldNotStatic); }
    public static Collection<Field> extensive_member_fields(Class cls, Consumer<Field> con) { return Lang.getExtensiceFields(cls, fieldNotStatic, con); }
    
    public static Collection<Field> extensive_static_fields(Class cls) { return Lang.getExtensiceFields(cls, fieldIsStatic); }
    public static Collection<Field> extensive_static_fields(Class cls, Consumer<Field> con) { return Lang.getExtensiceFields(cls, fieldIsStatic, con); }
    
    public static Collection<Field> extensive_final_fields(Class cls) { return Lang.getExtensiceFields(cls, fieldIsFinal); }
    public static Collection<Field> extensive_final_fields(Class cls, Consumer<Field> con) { return Lang.getExtensiceFields(cls, fieldIsFinal, con); }
    
    public static Collection<Field> extensive_not_final_fields(Class cls) { return Lang.getExtensiceFields(cls, fieldNotFinal); }
    public static Collection<Field> extensive_not_final_fields(Class cls, Consumer<Field> con) { return Lang.getExtensiceFields(cls, fieldNotFinal, con); }
    
    //<editor-fold defaultstate="collapsed" desc="Local:Core-Code"> 
    public static Collection<Field> getFields(Class clazz, Predicate<Field> pre) {
        Field[] fids = clazz.getFields();
        Collection<Field> c = new LinkedList<>();
        for(int i = 0; i < fids.length; i++) 
            if(pre.test(fids[i])) c.add(fids[i]);
        return c;
    }
    public static Collection<Field> getFields(Class clazz, Predicate<Field> pre, Consumer<Field> con) {
        Field[] fids = clazz.getFields();
        Collection<Field> c = new LinkedList<>();
        for(int i=0; i< fids.length;i++) 
            if(pre.test(fids[i])) { con.accept(fids[i]);c.add(fids[i]); }
        return c;
    }
    //</editor-fold>
    
    public static Collection<Field> getStaticFields(Class clazz) { return Lang.getFields(clazz, fieldIsStatic); }
    public static Collection<Field> getStaticFields(Class clazz, Predicate<Field> pre) { return Lang.getFields(clazz, fieldIsStatic.and(pre)); }
    public static Collection<Field> getStaticFields(Class clazz, Predicate<Field> pre, Consumer<Field> con) { return Lang.getFields(clazz, fieldIsStatic.and(pre), con); }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Method-Function">
    public static final Predicate methodIsStatic = (Predicate<Method>) (Method t) -> Modifier.isStatic(t.getModifiers());
    public static final Predicate methodNotStatic = (Predicate<Method>) (Method t) -> !Modifier.isStatic(t.getModifiers());
    public static final Predicate methodIsFinal = (Predicate<Method>) (Method t) -> Modifier.isFinal(t.getModifiers());
    public static final Predicate methodNotFinal = (Predicate<Method>) (Method t) -> !Modifier.isFinal(t.getModifiers());
    
    //<editor-fold defaultstate="collapsed" desc="Core-Code">
    /**
     * get all declared methods of a specified class, both its or its 
     * super classes'.
     * @param cls 
     * @return 
     */
    public static Collection<Method> extensive_methods(Class cls)  {
        Objects.requireNonNull(cls);
        Collection<Method> c = new LinkedList<>();
        for(; cls != null && cls != Object.class; cls = cls.getSuperclass()) {
            Method[] mths = cls.getDeclaredMethods();
            for (Method mth : mths) c.add(mth);
        }
        return c;
    }
     /**
     * get all declared Methods of a specified class, both its or its 
     * super classes'. for each Method you can use {@code Predicate pre} to check
     * and decide whether to add it to the Collection.
     * @param cls 
     * @param pre 
     * @return 
     */
    public static Collection<Method> extensive_methods(Class cls, Predicate<Method> pre) {
        Objects.requireNonNull(cls);
        Collection<Method> c = new LinkedList<>();
        for(;cls != null && cls != Object.class; cls = cls.getSuperclass()) {
            Method[] mths = cls.getDeclaredMethods();
            for (Method mth : mths) if (pre.test(mth)) c.add(mth);
        }
        return c;
    }
    /**
     * get all declared Methods of a specified class, both its or its 
     * super classes'. for each Method you can use {@code Predicate pre} to check
     * and decide whether to add it to the Collection. Besides, before you add
     * the field to the Collection, you can use con to process the field, like
     * {@code field.setAccessible(true);} and so on.
     * @param cls 
     * @param pre 
     * @param con 
     * @return 
     */
    public static Collection<Method> extensive_methods(Class cls, Predicate<Method> pre, Consumer<Method> con) {
        Objects.requireNonNull(cls);
        Collection<Method> c=new LinkedList<>();
        for(;cls != null && cls != Object.class; cls = cls.getSuperclass()) {
            Method[] mths = cls.getDeclaredMethods();
            for (Method mth : mths) if (pre.test(mth)) { con.accept(mth); c.add(mth); }
        }
        return c;
    }
    //</editor-fold>
    public static Collection<Method> exensive_member_methods(Class cls)  { return Lang.extensive_methods(cls, methodNotStatic); }
    public static Collection<Method> exensive_member_methods(Class cls, Consumer<Method> con) { return Lang.extensive_methods(cls, methodNotStatic, con); }
    
    public static Collection<Method> extensive_static_methods(Class cls) { return Lang.extensive_methods(cls, methodIsStatic); }
    public static Collection<Method> extensive_static_methods(Class cls, Consumer<Method> con) { return Lang.extensive_methods(cls, methodIsStatic, con); }
    
    public static Collection<Method> extensive_final_methods(Class cls) { return Lang.extensive_methods(cls, methodIsFinal); }
    public static Collection<Method> extensive_final_methods(Class cls, Consumer<Method> con) { return Lang.extensive_methods(cls, methodIsFinal, con); }
    
    public static Collection<Method> extensive_not_final_methods(Class cls) { return Lang.extensive_methods(cls, methodNotFinal); }
    public static Collection<Method> extensive_not_final_methods(Class cls, Consumer<Method> con) { return Lang.extensive_methods(cls, methodNotFinal, con); }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Package-Function">
    private static final ClassLoader DEF_CLL = ClassLoader.getSystemClassLoader();
    
    //<editor-fold defaultstate="collapsed" desc="Core: getClass">
    @Passed
    private static void addClassesFromFile(Set<Class> set, String pack, String path, ClassLoader cll) throws Exception
    {
        File dir=new File(path);  
        if(!dir.exists()||!dir.isDirectory()) return;  
        File[] files=dir.listFiles();
        String fileName=null;
        for(int i=0;i<files.length;i++)
        {  
            fileName=files[i].getName();
            if(files[i].isDirectory())
                addClassesFromFile(set, pack+'.'+fileName, files[i].getAbsolutePath(), cll);  
            else if(fileName.endsWith(".class"))
                set.add(cll.loadClass(pack+'.'+fileName.substring(0, fileName.length()-6)));//add class    
        }  
    }  
    @Passed
    private static void addClassesFromJar(Set<Class> set, URL url, String path, ClassLoader cll) throws Exception
    {
        JarFile jar=((JarURLConnection) url.openConnection()).getJarFile();  
        Enumeration<JarEntry> entries=jar.entries();  
        for(JarEntry entry;entries.hasMoreElements();)
        {  
            entry=entries.nextElement();  
            if(entry.isDirectory()) continue;
            String entryName=entry.getName(); 
            if(entryName.endsWith(".class")&&entryName.startsWith(path))
            {
                int len=entryName.length();
                if(entryName.charAt(0)=='/') entryName=entryName.substring(1);
                if(entryName.charAt(len-1)=='/') entryName=entryName.substring(0, len-1);
                set.add(cll.loadClass(entryName.substring(0, len-6).replace('/', '.')));//class full path
            }
        }  
    }
    @Passed
    public static void getClasses(Set<Class> set, String pack, ClassLoader cll) throws Exception
    {  
        String path=pack.replace('.', '/');
        Enumeration<URL> dirs=cll.getResources(path);  
        
        for(URL url;dirs.hasMoreElements();) 
        {  
            url=dirs.nextElement();  
            switch(url.getProtocol())
            {
                case "file":Lang.addClassesFromFile(set, pack, URLDecoder.decode(url.getFile(), "UTF-8"), cll);break;
                case "jar":Lang.addClassesFromJar(set, url, path, cll);break;
            }
        }  
    }  
    public static Set<Class> getClasses(String pack, ClassLoader cll) throws Exception
    {  
        Set<Class> set=new HashSet<>();  
        Lang.getClasses(set, pack, cll);
        return set;  
    }  
    public static Set<Class> getClasses(String[] packs, ClassLoader cll) throws Exception
    {
        Set<Class> set=new HashSet<>();
        for(int i=0;i<packs.length;i++) Lang.getClasses(set, packs[i], cll);
        return set;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Core-Code:getClassNeat">
    @Passed
    private static void addClassesFromFileNeat(Set<Class> set, String pack, String path, ClassLoader cll) throws Exception
    {
        File dir=new File(path);  
        if(!dir.exists()||!dir.isDirectory()) return;  
        File[] files=dir.listFiles();
        String fileName=null;
        for(int i=0;i<files.length;i++)
        {  
            fileName=files[i].getName();
            if(files[i].isDirectory())
                addClassesFromFileNeat(set, pack+'.'+fileName, files[i].getAbsolutePath(), cll);  
            else if(fileName.endsWith(".class"))
            {
                fileName=fileName.substring(0, fileName.length()-6);
                int index=fileName.lastIndexOf('$');
                if(index!=-1) fileName=fileName.substring(0, index);
                set.add(cll.loadClass(pack+'.'+fileName));//add class    
            }
        }  
    }  
    @Passed
    private static void addClassesFromJarNeat(Set<Class> set, URL url, String path, ClassLoader cll) throws Exception
    {
        JarFile jar=((JarURLConnection) url.openConnection()).getJarFile();  
        Enumeration<JarEntry> entries=jar.entries();  
        for(JarEntry entry;entries.hasMoreElements();)
        {  
            entry=entries.nextElement();  
            if(entry.isDirectory()) continue;
            String entryName=entry.getName(); 
            if(entryName.endsWith(".class")&&entryName.startsWith(path))
            {
                int len=entryName.length();
                if(entryName.charAt(0)=='/') entryName=entryName.substring(1);
                if(entryName.charAt(len-1)=='/') entryName=entryName.substring(0, len-1);
                
                entryName=entryName.substring(0, len-6).replace('/', '.');
                int index=entryName.lastIndexOf('$');
                if(index!=-1) entryName=entryName.substring(0, index);
                set.add(cll.loadClass(entryName));//class full path
            }
        }  
    }
    @Passed
    public static void getClassesNeat(Set<Class> set, String pack, ClassLoader cll) throws Exception
    {  
        String path=pack.replace('.', '/');
        Enumeration<URL> dirs=cll.getResources(path);  
        
        for(URL url;dirs.hasMoreElements();) 
        {  
            url=dirs.nextElement();  
            switch(url.getProtocol())
            {
                case "file":Lang.addClassesFromFileNeat(set, pack, URLDecoder.decode(url.getFile(), "UTF-8"), cll);break;
                case "jar":Lang.addClassesFromJarNeat(set, url, path, cll);break;
            }
        }  
    }  
    @Passed
    public static Set<Class> getClassesNeat(String pack, ClassLoader cll) throws Exception
    {  
        Set<Class> set=new HashSet<>();  
        Lang.getClassesNeat(set, pack, cll);
        return set;  
    }  
    public static void makeClassSetNeat(Set<Class> set, ClassLoader cll) throws ClassNotFoundException 
    {
        Set<String> names=new HashSet<>();
        int index=0;
        String name=null;
        for(Class cls:set)
        {
            name=cls.getName();
            index=name.lastIndexOf('$');
            if(index!=-1) name=name.substring(0, index);
            names.add(name);
        }
        set.clear();
        for(String nam:names) set.add(cll.loadClass(nam));
    }
    //</editor-fold>
    public static Set<Class> getClasses(String pack) throws Exception { return Lang.getClasses(pack, DEF_CLL); }
    public static Set<Class> getClassesNeat(String pack) throws Exception
    {
        return Lang.getClassesNeat(pack, DEF_CLL);
    }
    public static Set<Class> getClasses(String[] packs) throws Exception
    {
        Set<Class> set=new HashSet<>();
        for(int i=0;i<packs.length;i++) Lang.getClasses(set, packs[i], DEF_CLL);
        return set;
    }
    public static Set<Class> getClassesNeat(String[] packs) throws Exception
    {
        Set<Class> set=new HashSet<>();
        for(int i=0;i<packs.length;i++) Lang.getClassesNeat(set, packs[i], DEF_CLL);
        return set;
    }
    public static void makeClassSetNeat(Set<Class> set) throws ClassNotFoundException 
    {
        Lang.makeClassSetNeat(set, DEF_CLL);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Clone-Function">
    //<editor-fold defaultstate="collapsed" desc="function: serialize">
    public static byte[] serialize(Object o) {
        byte[] arr = null;
        ByteArrayOutputStream bout = null;
        ObjectOutputStream oos = null;
        try {
            bout = new ByteArrayOutputStream();
            oos = new ObjectOutputStream(bout);
            oos.flush();
            arr = bout.toByteArray();
        }
        catch(IOException e) { throw new RuntimeException(e); }
        finally { try {
            if (oos != null) oos.close();
            if (bout != null) bout.close();
        } catch(IOException e) { throw new RuntimeException(e); } }
        return arr;
    }
    
    public static <T extends Serializable> T deserialize(byte[] arr) {
        T result = null;
        ByteArrayInputStream bin = null;
        ObjectInputStream ois = null;
        try {
            bin = new ByteArrayInputStream(arr);
            ois = new ObjectInputStream(bin);
            result = (T) ois.readObject();
        }
        catch(IOException | ClassNotFoundException e) { throw new RuntimeException(e); }
        finally { try {
            if (ois != null) ois.close();
            if (bin != null) bin.close();
        } catch(IOException e) { throw new RuntimeException(e); } }
        return result;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="function: clone">
    public static <T extends Serializable> T clone(Object o) {
        T result = null;
        ByteArrayOutputStream bout = null;
        ObjectOutputStream oos = null;
        ByteArrayInputStream bin = null;
        ObjectInputStream ois = null;
        try {
            bout = new ByteArrayOutputStream(1024);
            oos = new ObjectOutputStream(bout);
            oos.writeObject(o);
            oos.flush();
            bin = new ByteArrayInputStream(bout.toByteArray());
            ois = new ObjectInputStream(bin);
            result = (T) ois.readObject();
        } 
        catch(IOException | ClassNotFoundException e) { throw new RuntimeException(e); }
        finally { try {
            if (ois != null) ois.close();
            if (bin != null) bin.close();
            if (oos != null) oos.close();
            if (bout != null) bout.close();
        } catch(IOException e) { throw new RuntimeException(e); } }
        return result;
    }
    
    public static <T extends Serializable> List<T> clone(Object o, int n) {
        List<T> result = null;
        ByteArrayOutputStream bout = null;
        ObjectOutputStream oos = null;
        ByteArrayInputStream bin = null;
        ObjectInputStream ois = null;
        try {
            bout = new ByteArrayOutputStream(1024);
            oos = new ObjectOutputStream(bout); 
            oos.writeObject(o);
            oos.flush();
            byte[] arr = bout.toByteArray();
            
            result = new LinkedList<>();
            for(int i=0;i<n;i++) {
                oos.writeObject(o);
                bin = new ByteArrayInputStream(arr);
                ois = new ObjectInputStream(bin);
                result.add((T) ois.readObject());
            }
        }
        catch(IOException | ClassNotFoundException e) { throw new RuntimeException(e); }
        finally { try {
            if (ois != null) ois.close();
            if (bin != null) bin.close();
            if (oos != null) oos.close();
            if (bout != null) bout.close();
        } catch(IOException e) { throw new RuntimeException(e); } }
        return result;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Convert-Function">
    public static String converterDetail()  {
        StringBuilder sb = new StringBuilder(512);
        sb.append("Map: class->Converter = {"); Vector.appendLn(sb, CLASS_CONVERTER_MAP, "\t\n"); sb.append("\n}\n");
        sb.append("Map: name->Converter = {");  Vector.appendLn(sb, NAME_CONVERTER_MAP, "\t\n");  sb.append("\n}\n");
        return sb.toString();
    }
    
    public static Converter converter(Class<?> cls)  {
        Converter con = CLASS_CONVERTER_MAP.get(cls);
        if(con == null) throw new RuntimeException("There is no matched Converter:" + cls);
        return con;
    }
    
    public static Converter converter(String name) {
        Converter con = NAME_CONVERTER_MAP.get(name);
        if(con == null) throw new RuntimeException("There is no matched Converter:" + name);
        return con;
    }
    
    public static <T> T convert(String str, Class<?> cls) throws Exception {
        Converter con = CLASS_CONVERTER_MAP.get(cls);
        if(con == null) throw new RuntimeException("There is no matched Converter for class:" + cls);
        return con.convert(str);
    }
    
    public static <T> T convert(String str, String cls_name) throws Exception {
        Converter con = NAME_CONVERTER_MAP.get(cls_name);
        if(con == null) throw new RuntimeException("There is no matched Converter for class:" + cls_name);
        return con.convert(str);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="toString-Function">
    public static String stringerDetail() {
        StringBuilder sb = new StringBuilder(512);
        sb.append("Map: class->Stringer = {");
        Vector.appendLn(sb, CLASS_STRINGER_MAP, "\t\n");
        sb.append("\n}\n");
        return sb.toString();
    }

    public static Stringer<?> stringer(Class<?> cls)  {
        Stringer<?> str = CLASS_STRINGER_MAP.get(cls);
        if(str == null) throw new RuntimeException("There is no matched Stringer:" + cls);
        return str;
    }
    
    public static String toString(Object val, Class<?> cls) {
        Stringer str = CLASS_STRINGER_MAP.get(cls);
        if (str != null) return str.process(val);//find the corrosponding stringer
        if ((cls = Lang.is_array(cls)) != null)  {
            if ((cls = Lang.is_array(cls)) != null) return Matrix.toString((Object[][])val);
            return Vector.toString((Object[])val);
        }
        return val.toString();
    }
    
    public static String toString(Object val) { return (val == null? "NULL" : Lang.toString(val, val.getClass())); }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Print-Function">
    private static PrintStream DEF_OUT=System.out;
    public static synchronized void setDefaultPrintStream(PrintStream out) {DEF_OUT=out;}
    public static PrintStream getDefaultPrintStream(){return DEF_OUT;}
    
    @Passed
    public static String printerDetail()
    {
        StringBuilder sb=new StringBuilder();
        
        sb.append("Map: class->Printer = {");
        Vector.appendLn(sb, CLASS_PRINTER_MAP, "\t\n");
        sb.append("\n}\n");
        
        return sb.toString();
    }
    @Passed
    public static Printer getPrinter(Class clazz) 
    {
        Printer pr=CLASS_PRINTER_MAP.get(clazz);
        if(pr==null) throw new RuntimeException("There is no matched Printer:"+clazz);
        return pr;
    }
    @Passed
    public static void println(Object val, Class clazz)
    {
        Printer pr=CLASS_PRINTER_MAP.get(clazz);
        if(pr!=null) {pr.println(DEF_OUT, val);return;}
        if((clazz=Lang.is_array(clazz))!=null)
        {
            if((clazz=Lang.is_array(clazz))!=null) {Matrix.println(DEF_OUT, (Object[][])val);return;}
            Vector.println(DEF_OUT, (Object[])val);return;
        }
        DEF_OUT.println(val);
    }
    public static void println(Object val)
    {
        if(val==null) {DEF_OUT.println(val);}
        else Lang.println(val, val.getClass());
    }
    @Passed
    public static void zprintln(Object... args)
    {
        if(args==null) {DEF_OUT.println(NULL);return;}
        for(int i=0;i<args.length;i++)
        {
            DEF_OUT.print("args ["+i+"] :");
            Lang.println(args[i]);
            DEF_OUT.println();
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="StringBuilder: extend">
    private static Field StringBuilder_value;
    private static Field String_value;
    static {
        try {
            Class cls = StringBuilder.class.getSuperclass();
            StringBuilder_value = cls.getDeclaredField("value");
            StringBuilder_value.setAccessible(true);
            
            String_value = String.class.getDeclaredField("value");
            String_value.setAccessible(true);
        }
        catch(NoSuchFieldException | SecurityException e) { e.printStackTrace(); }
    }
    
    public static final char[] getChars(StringBuilder sb) throws Exception { return (char[]) StringBuilder_value.get(sb); }
    public static final char[] getChars(String sb) throws Exception { return (char[]) String_value.get(sb); }
    //</editor-fold>
}
