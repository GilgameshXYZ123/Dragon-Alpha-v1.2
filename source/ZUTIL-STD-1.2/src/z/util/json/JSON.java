/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.json;

import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;
import z.util.json.annotation.JSONEntity;
import z.util.lang.Lang;
import static z.util.lang.Lang.NULL;

/**
 *
 * @author dell
 */
public class JSON 
{
    //<editor-fold defaultstate="collapsed" desc="static class JSONParserMap">
    static JSONParser elementParser = new JSONParser(){
        @Override
        public void append(StringBuilder sb, Object obj) throws Exception  {
            sb.append('\"').append(obj).append('\"');
        }
    };
    
    static JSONParser mapParser = new JSONParser() {
        @Override
        public void append(StringBuilder sb, Object obj) throws Exception  {
            Set<? extends Entry> kvs = ((Map) obj).entrySet();
            sb.append('{');
            int index = 0; Object val;
            for(Entry kv:kvs) {
                if(index!=0) sb.append(',');
                sb.append('\"').append(kv.getKey()).append("\":");
                val = kv.getValue();
                map.getParserOrLoad(val.getClass()).append(sb, val);
                index++;
            }
            sb.append('}');
        }
    };
    
    public static class JSONParserMap extends HashMap<Class,JSONParser> {
        public <T extends JSONParser> T getParserOrLoad(Class key) throws Exception  {
            JSONParser jp = this.get(key);
            if(jp == null) {
                boolean all = key.getAnnotation(JSONEntity.class)!=null;
                if(all) jp = new JSONClassParser(key);
                else if (Lang.is_sub_cls(key, Map.class)) jp=JSON.mapParser;
                else jp = JSON.elementParser;
                this.put(key, jp);
            }
            return (T) jp;
        }
    }
    //</editor-fold>
    
    static JSONParserMap map=new JSONParserMap();
    
    //<editor-fold defaultstate="collapsed" desc="Operators:toJSON">
    public static String toJSON(Object obj) throws Exception {
        if(obj == null) return NULL;
        StringBuilder sb = new StringBuilder(128);
        map.getParserOrLoad(obj.getClass()).append(sb, obj);
        return sb.toString();
    }

    public static String toJSON(Object obj, Class clazz) throws Exception {
         if(obj == null) return NULL;
        StringBuilder sb = new StringBuilder(128);
        map.getParserOrLoad(clazz).append(sb ,obj);
        return sb.toString();
    }

    public static <T extends Object> String toJSON(T[] arr) throws Exception  {
        if(arr == null) return NULL;
        if(arr.length == 0) return "[]";
        
        StringBuilder sb = new StringBuilder(128);
        map.getParserOrLoad(arr[0].getClass())
                .append(sb.append('['), arr[0]);
        
        for(int i=1;i<arr.length;i++)
            map.getParserOrLoad(arr[i].getClass())
                    .append(sb.append(",\n"), arr[i]);
        
        return sb.append(']').toString();
    }
    
    public static <T extends Object> String toJSON(T[] arr, Class clazz) throws Exception {
        if(arr == null) return "null";
        if(arr.length == 0) return "[]";
        
        JSONParser jp = map.getParserOrLoad(clazz);
        StringBuilder sb = new StringBuilder(128);
        jp.append(sb.append('['), arr[0]);
        
        for(int i=1; i<arr.length; i++) jp.append(sb.append(",\n"), arr[i]);        
        return  sb.append(']').toString();
    }
    
    public static <T extends Object> String toJSON (Collection<T> obj) throws Exception {
        if(obj == null) return NULL;
        if(obj.isEmpty()) return "[]";
        
        StringBuilder sb = new StringBuilder(128);
        Iterator<T> iter=obj.iterator();
        
        T val=iter.next();
        map.getParserOrLoad(val.getClass()).append(sb.append('['), val);
        
        while(iter.hasNext()) {
            val = iter.next();
            map.getParserOrLoad(val.getClass()).append(sb.append(",\n"), val);
        }
        
        return sb.append(']').toString();
    }
   
    public static <T extends Object> String toJSON (Collection<T> obj, Class clazz) throws Exception {
        if(obj == null) return NULL;
        if(obj.isEmpty()) return "[]";
        
        StringBuilder sb = new StringBuilder(128);
        Iterator<T> iter = obj.iterator();
        
        JSONParser jp = map.getParserOrLoad(clazz);
        jp.append(sb.append('['), iter.next());
        
        while(iter.hasNext()) jp.append(sb.append(",\n"), iter.next());
        
        return sb.append(']').toString();
    }
    //</editor-fold>
}
