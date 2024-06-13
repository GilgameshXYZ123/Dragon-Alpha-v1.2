/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.db.jdbc.query;

import z.util.db.jdbc.query.annotation.PrimaryKey;
import z.util.db.jdbc.query.annotation.Column;
import z.util.db.jdbc.query.annotation.Table;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.lang.reflect.Modifier;
import z.util.ds.ZEntry;

import z.util.lang.annotation.Passed;

/**
 *
 * @author dell
 */
class QueryParser 
{
   
    //<editor-fold defaultstate="collapsed" desc="static class ColName-Field">
    final static class ColumnField extends ZEntry<String,Field> 
    {
        ColumnField() {}
        ColumnField(String key, Field value)
        {
            super(key, value);
        }
    }
    //</editor-fold>
    
    Constructor con;
    ColumnField[] cfs;
    ColumnField pk;//@primaryKey
    HashMap<String, ColumnField> hcfs;
    String tableName;
    
    public QueryParser(Class clazz) throws Exception
    {
        Objects.requireNonNull(clazz, "Class");
        this.con=clazz.getConstructor();
        //table name------------------------------------------------------------
        Table tableAno=(Table) clazz.getAnnotation(Table.class);
        tableName=(tableAno!=null? tableAno.value():clazz.getSimpleName());
       
        //fields----------------------------------------------------------------
        Class cls=clazz;
        Field[] fids;
        LinkedList<ColumnField> list=new LinkedList<>();
        Column colAno;
        String column;
        while(cls!=null&&cls!=Object.class)
        {
            fids=cls.getDeclaredFields();
            for (Field fid : fids) {
                if (Modifier.isStatic(fid.getModifiers())) continue;
                colAno = fid.getAnnotation(Column.class);
                column = (colAno!=null ? colAno.value() : fid.getName());
                fid.setAccessible(true);
                list.add(new ColumnField(column, fid));
            }
            cls=cls.getSuperclass();
        }
        
        cfs=new ColumnField[list.size()];
        hcfs=new HashMap<>();
        int index=0;
        for(ColumnField v:list) 
        {
            cfs[index++]=v;
            hcfs.put(v.key, v);
        }
        list.clear();
        
        //primary key-----------------------------------------------------------
        pk=new ColumnField();
        PrimaryKey pkAno=(PrimaryKey) clazz.getAnnotation(PrimaryKey.class);
        pk.key=(pkAno!=null? pkAno.value():"id");
        for (ColumnField cf : cfs) 
            if (cf.key.equals(pk.key)) 
            {
                pk.value = cf.value;
                break;
            }
    }
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
    @Override
    public String toString()
    {
        StringBuilder sb=new StringBuilder();
        sb.append("QueryParser = {");
        sb.append("\n\t table = ").append(tableName);
        sb.append("\n\t primaryKey = ").append(pk);
        sb.append("\n\t columns = {");
        for(int i=0;i<cfs.length;i++) sb.append("\n\t\t").append(cfs[i]);
        sb.append("\n\t}\n}");
        return sb.toString();
    }
    private ColumnField[] getSpecColumns(String[] colName)
    {
        ColumnField[] dcfs=new ColumnField[colName.length];
        for(int i=0;i<colName.length;i++)
            dcfs[i]=hcfs.get(colName[i]);
        return dcfs;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Object:Parser">
    @Passed
    public <T> T parse(Object[] args) throws Exception//pase
    {
        if(args==null) throw new NullPointerException();
        T r=(T) con.newInstance();
        if(args.length!=cfs.length) throw new RuntimeException("args.length!=cfs.length");
        for(int i=0;i<cfs.length;i++)
            cfs[i].value.set(r, args[i]);
        return r;
    }
    @Passed
    public <T> LinkedList<T> parseBatch(List<Object[]> largs) throws Exception//pase
    {
        if(largs==null) throw new NullPointerException();
        LinkedList<T> list=new LinkedList<>();
        Object r=null;
        int i;
        for(Object[] args:largs)
        {
            r=con.newInstance();
            for(i=0;i<cfs.length;i++)
                cfs[i].value.set(r, args[i]);
            list.add((T) r);
        }
        return list;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Select:Parser">
    @Passed
    public <T> T parse(ResultSet rs) throws Exception
    {
        if(!rs.next()) return null;
        T r=(T) con.newInstance();
        for (ColumnField cf : cfs) 
            cf.value.set(r, rs.getObject(cf.key));
        return r;
    }
    @Passed
    public <T> T parse(ResultSet rs, String[] colName) throws Exception
    {
        if(!rs.next()) return null;
        ColumnField[] dcfs=this.getSpecColumns(colName);
        T r=(T) con.newInstance();
        for(int i=0;i<dcfs.length;i++)
            cfs[i].value.set(r, rs.getObject(dcfs[i].key));
        return r;
    }
    @Passed
    public <T> LinkedList<T> parseBatch(ResultSet rs) throws Exception//pase
    {
        LinkedList<T> list=new LinkedList<>();
        Object r;
        int i;
        while(rs.next())
        {
            r=con.newInstance();
            for(i=0;i<cfs.length;i++)
               cfs[i].value.set(r, rs.getObject(cfs[i].key));
            list.add((T) r);
        }
        return list;
    }
    @Passed
    public <T> LinkedList<T> parseBatch(ResultSet rs, String[] colName) throws Exception//pase
    {
        ColumnField[] dcfs=this.getSpecColumns(colName);
        LinkedList<T> list=new LinkedList<>();
        Object r;
        int i;
        while(rs.next())
        {
            r=con.newInstance();
            for(i=0;i<dcfs.length;i++)
                dcfs[i].value.set(r, rs.getObject(dcfs[i].key));
            list.add((T) r);
        }
        return list;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Insert:Setter">
    //insert into <table>(<columns...>) values(?,?,?);
    @Passed
    public void setInsert(PreparedStatement pst, Object val) throws Exception
    {
        for(int i=0;i<cfs.length;i++) pst.setObject(i+1, cfs[i].value.get(val));
        pst.addBatch();
    }
    @Passed
    public void setInsert(PreparedStatement pst,Object val, String[] colName) throws Exception
    {
        ColumnField[] dcfs=this.getSpecColumns(colName);
        for(int i=0;i<dcfs.length;i++) pst.setObject(i+1, dcfs[i].value.get(val));
        pst.addBatch();
    }
    @Passed
    public void setInsertBatch(PreparedStatement pst, Collection batch) throws Exception
    {
        int i;
        for(Object val:batch)
        {
            for(i=0;i<cfs.length;i++) pst.setObject(i+1, cfs[i].value.get(val));
            pst.addBatch();
        }
    }
    @Passed
    public void setInsertBatch(PreparedStatement pst, Collection batch, String[] colName) throws Exception
    {
        ColumnField[] dcfs=this.getSpecColumns(colName);
        int i;
        for(Object val:batch)
        {
            for(i=0;i<dcfs.length;i++) pst.setObject(i+1, dcfs[i].value.get(val));
            pst.addBatch();
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Update:Setter">
    //update <table> set{columname=value} where <pk=?> 
    @Passed
    public void setUpdate(PreparedStatement pst, Object val) throws Exception
    {
        int i;
        for(i=0;i<cfs.length;i++)
            pst.setObject(i+1, cfs[i].value.get(val));
        pst.setObject(i+1, pk.value.get(val));
        pst.addBatch();
    }
    @Passed
    public void setUpdate(PreparedStatement pst,Object val, String[] colName) throws Exception
    {
        ColumnField[] dcfs=this.getSpecColumns(colName);
        int i;
        for(i=0;i<dcfs.length;i++)
            pst.setObject(i+1, dcfs[i].value.get(val));
        pst.setObject(i+1, pk.value.get(val));
        pst.addBatch();
    }
    @Passed
    public void setUpdateBatch(PreparedStatement pst, Collection batch) throws Exception
    {
        int i;
        for(Object val:batch)
        {
            for(i=0;i<cfs.length;i++)
                pst.setObject(i+1, cfs[i].value.get(val));
            pst.setObject(i+1, pk.value.get(val));
            pst.addBatch();
        }
    }
    @Passed
    public void setUpdateBatch(PreparedStatement pst, Collection batch, String[] colName) throws Exception
    {
        ColumnField[] dcfs=this.getSpecColumns(colName);
        int i;
        for(Object val:batch)
        {
            for(i=0;i<dcfs.length;i++)
                pst.setObject(i+1, dcfs[i].value.get(val));
            pst.setObject(i+1, pk.value.get(val));
            pst.addBatch();
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Delete:Setter">   
    //delete from <table> where <pk=?>
    @Passed
    public void setDelete(PreparedStatement pst, Object val) throws Exception
    {
        pst.setObject(1, pk.value.get(val));
        pst.addBatch();
    }
    @Passed
    public void setDeleteBatch(PreparedStatement pst, Collection batch) throws Exception
    {
        for(Object val:batch)
        {
            pst.setObject(1, pk.value.get(val));
            pst.addBatch();
        }
    }
    //</editor-fold>
}
