package org.CT417Assignment1;

import org.joda.time.DateTime;

import java.lang.reflect.Array;
import java.util.ArrayList;

public class Student
{
    private String name;
    private int age;
    private DateTime dob;
    private static long id = 0;
    private String username;
    private ArrayList<Course> courses;
    private ArrayList<Module> modules;

    public Student(String name, int age, DateTime dob, ArrayList<Course> courses, ArrayList<Module> modules)
    {
        this.name = name;
        this.age = age;
        this.dob = dob;
        this.id++;
        this.courses = courses;
        this.modules = modules;
        String nameWithoutSpaces = this.name.replaceAll("\\s+","");
        this.username = nameWithoutSpaces + this.age;
    }

    public String GetName()
    {
        return this.name;
    }

    public int GetAge()
    {
        return this.age;
    }

    public String GetUsername()
    {
        String nameWithoutSpaces = this.name.replaceAll("\\s+","");
        this.username = nameWithoutSpaces + this.age;

        return this.username;
    }

    public DateTime GetDateOfBirth()
    {
        return this.dob;
    }

    public long GetId()
    {
        return this.id;
    }

    public ArrayList<Course> GetCourses()
    {
        return this.courses;
    }

    public ArrayList<Module> GetModules()
    {
        return this.modules;
    }

}
