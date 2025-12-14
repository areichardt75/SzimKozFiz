import tkinter as tk
from tkinter import messagebox

class BeolvasoAblak:
    def __init__(self):
        self.eredmeny = None
        
    def megjelenit(self, cim="Adatbevitel", uzenet="Kérem adja meg az értéket:"):
        """Megjeleníti a beolvasó ablakot és visszaadja az eredményt"""
        ablak = tk.Tk()
        ablak.title(cim)
        ablak.geometry("400x150")
        
        # Címke
        label = tk.Label(ablak, text=uzenet, font=("Arial", 11))
        label.pack(pady=10)
        
        # Beviteli mező
        bemeneti_mezo = tk.Entry(ablak, width=40, font=("Arial", 11))
        bemeneti_mezo.pack(pady=10)
        bemeneti_mezo.focus()
        
        def ok_gomb_kattintas():
            self.eredmeny = bemeneti_mezo.get()
            if self.eredmeny:
                ablak.destroy()
            else:
                messagebox.showwarning("Figyelem", "Kérem adjon meg értéket!")
        
        def megse_gomb_kattintas():
            self.eredmeny = None
            ablak.destroy()
        
        # Enter billentyű kezelése
        bemeneti_mezo.bind('<Return>', lambda e: ok_gomb_kattintas())
        
        # Gombok
        gomb_keret = tk.Frame(ablak)
        gomb_keret.pack(pady=10)
        
        ok_gomb = tk.Button(gomb_keret, text="OK", command=ok_gomb_kattintas, 
                           width=10, bg="#4CAF50", fg="white")
        ok_gomb.pack(side=tk.LEFT, padx=5)
        
        megse_gomb = tk.Button(gomb_keret, text="Mégse", command=megse_gomb_kattintas,
                              width=10, bg="#f44336", fg="white")
        megse_gomb.pack(side=tk.LEFT, padx=5)
        
        # Ablak középre igazítása
        ablak.update_idletasks()
        x = (ablak.winfo_screenwidth() // 2) - (ablak.winfo_width() // 2)
        y = (ablak.winfo_screenheight() // 2) - (ablak.winfo_height() // 2)
        ablak.geometry(f"+{x}+{y}")
        
        ablak.mainloop()
        return self.eredmeny


# ===== HASZNÁLATI PÉLDÁK =====

# 1. Egyszerű használat
def pelda1():
    beolvaso = BeolvasoAblak()
    nev = beolvaso.megjelenit("Név bekérése", "Adja meg a nevét:")
    
    if nev:
        print(f"A beolvasott név: {nev}")
    else:
        print("Nem adott meg nevet (Mégse gomb)")


# 2. Több adat bekérése egymás után
def pelda2():
    beolvaso = BeolvasoAblak()
    
    nev = beolvaso.megjelenit("Adatok", "Név:")
    if not nev:
        return
    
    kor = beolvaso.megjelenit("Adatok", "Életkor:")
    if not kor:
        return
    
    varos = beolvaso.megjelenit("Adatok", "Város:")
    if not varos:
        return
    
    print(f"\nBeolvasott adatok:")
    print(f"Név: {nev}")
    print(f"Életkor: {kor}")
    print(f"Város: {varos}")


# 3. Adat mentése fájlba
def pelda3():
    beolvaso = BeolvasoAblak()
    szoveg = beolvaso.megjelenit("Mentés", "Mit szeretne menteni?")
    
    if szoveg:
        with open('beolvasott_adat.txt', 'w', encoding='utf-8') as f:
            f.write(szoveg)
        print(f"'{szoveg}' sikeresen mentve!")


# Válassz egy példát a futtatáshoz:
if __name__ == "__main__":
    print("Válassz példát:")
    print("1 - Egyszerű név bekérés")
    print("2 - Több adat bekérése")
    print("3 - Adat bekérése és mentése")
    
    pelda1()  # Változtasd pelda2()-re vagy pelda3()-ra a másik példához
