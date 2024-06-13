/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.factory;

import java.io.Serializable;
import java.lang.reflect.Constructor;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import z.util.xml.XML;
import z.util.lang.annotation.Passed;

/**
 *
 * @author dell
 */
public class Resource implements Serializable
{
    String name;
    Class clazz;
    Constructor cons;
    Meta columns;
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
        //getter and setter-----------------------------------------------------
    public String getName() 
    {
        return name;
    }
    public void setName(String name) 
    {
        this.name = name;
    }
    public Class getClazz() 
    {
        return clazz;
    }
    public void setClazz(Class clazz) 
    {
        this.clazz = clazz;
    }
    public Meta getColumns() 
    {
        return columns;
    }
    public void setColumns(Meta columns) 
    {
        this.columns = columns;
    }
    public Constructor getCons() throws Exception
    {
        if(cons==null) cons=clazz.getConstructor();
        return cons;
    }
    
    public void append(StringBuilder sb)
    {
         sb.append(name).append(" = ").append(clazz);
    }
    @Override
    public String toString()
    {
        return name+" = "+clazz;
    }
    public void appendDetail(StringBuilder sb)
    {
        sb.append("Resource = ").append(name).append(" {");
        sb.append("\n\tclass = ").append(clazz);
        sb.append("\n\tcolumns = ").append(columns);
        sb.append("\n}");
    }
    public String toDetailString()
    {
        StringBuilder sb=new StringBuilder();
        this.appendDetail(sb);
        return sb.toString();
    }
    //</editor-fold>
    @Passed
    public static Resource valueOf(Node resource, String pack) throws Exception
    {
        Resource rs=new Resource();
        NamedNodeMap attr=resource.getAttributes();
        rs.name=XML.getAttributeNoNull(attr, "name");
        rs.clazz=Class.forName(pack+XML.getAttributeNoNull(attr, "class"));
        rs.columns=Meta.valueOf(resource.getChildNodes());
        return rs;
    }
}
