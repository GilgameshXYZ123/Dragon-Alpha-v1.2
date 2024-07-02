/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.json;

import z.util.json.annotation.JSONField;
import java.io.Serializable;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import z.util.ds.linear.ZLinkedList;
import static z.util.json.JSON.map;
import z.util.lang.Lang;
import z.util.math.vector.Vector;

/**
 *
 * @author dell
 */
public class JSONClassParser extends JSONParser
{
    //<editor-fold defaultstate="collapsed" desc="ColName-Field">
    final static class CF implements Serializable
    {
        //columns---------------------------------------------------------------
        String column;
        Field field;
        JSONParser parser;

        //functions-------------------------------------------------------------
        public CF() {}
        public CF(String column, Field field, JSONParser parser)
        {
            this.column = column;
            this.field = field;
            this.parser = parser;
        }
        @Override
        public String toString()
        {
            return column+" = "+field;
        }
    }
    //</editor-fold>
    Constructor con;
    CF[] cfs;
    
    public JSONClassParser(Class clazz) throws Exception {
        if(clazz==null) throw new NullPointerException();
        con=clazz.getConstructor();
        
        //fields----------------------------------------------------------------
        Field[] fids;
        ZLinkedList<CF> list=new ZLinkedList<>();
        JSONField jfield;
        String column;
        for(Class cls=clazz,flcls;cls!=null&&cls!=Object.class;cls=cls.getSuperclass())
        {
            fids=cls.getDeclaredFields();
            for (Field fid : fids) 
            {
                if(Modifier.isStatic(fid.getModifiers())) continue;
                //on the field be annotationed by 'JSONFiled',it can be jsonified
                jfield = fid.getAnnotation(JSONField.class);
                column=(jfield!=null? jfield.value():fid.getName());
                if(column==null) continue;
                fid.setAccessible(true);
                flcls = fid.getType();
                list.add(new CF(column, fid, Lang.is_elem_type(flcls)? null:map.getParserOrLoad(flcls)));
            }
        }
        cfs=new CF[list.number()];
        int index=0;
        for(CF v:list) cfs[index++]=v;
        list.clear();
    }
    
    @Override
    public String toString() {
        StringBuilder sb=new StringBuilder();
        sb.append("JSONParser = {");
        sb.append("\n\t columns = {");
        Vector.append(sb, "\n\t\t", cfs);
        sb.append("\n\t}\n}");
        return sb.toString();
    }
    
    private void append(StringBuilder sb, CF cf,Object obj) throws Exception {
        Object val=cf.field.get(obj);
        if(val==null) sb.append("null");
        else if(cf.parser==null) sb.append('\"').append(val).append('\"');//if the field is an element type
        else cf.parser.append(sb, val);//if the field is a complex type
    }
    
    @Override
    public void append(StringBuilder sb, Object obj) throws Exception {
        //for the start column--------------------------------------------------
        sb.append("{\"").append(cfs[0].column).append("\":");
        append(sb, cfs[0], obj);
        
        //for the next----------------------------------------------------------
        for(int i=1;i<cfs.length;i++)
        {
            sb.append(",\"").append(cfs[i].column).append("\":");
            append(sb, cfs[i], obj);
        }
        sb.append('}');
    }
}
