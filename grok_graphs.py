import pandas as pd
import matplotlib.pyplot as plt

def plot_game(csv_path, title, fair_cents):
    df = pd.read_csv(csv_path)
    df['ts'] = pd.to_datetime(df['ts_utc'])
    df = df.sort_values('ts')
    
    plt.figure(figsize=(14, 6))
    plt.plot(df['ts'], df['yes_best_bid'], label='YES bid', lw=1.8)
    plt.axhline(fair_cents, color='green', ls='--', label=f'Your fair ({fair_cents}¢)')
    
    # Entries
    entries = df[(df['open_qty'].shift(1) == 0) & (df['open_qty'] > 0)]
    plt.scatter(entries['ts'], entries['yes_imp_ask'], color='blue', s=80, marker='^', label='Entry')
    
    # Locks
    locks = df[df['pairs_count'].diff() > 0]
    plt.scatter(locks['ts'], locks['yes_best_bid'], color='purple', s=80, marker='o', label='Locked pair')
    
    plt.title(title)
    plt.ylabel('YES price (cents)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_game("Utah St. at New Mexico_KXNCAAMBGAME-26FEB04USUUNM-USU_20260205_040959.csv", 'Utah St (won) — big drawdown then 4 locks', 54)
plot_game("Iowa at Washington_KXNCAAMBGAME-26FEB04IOWAWASH-IOWA_20260205_040959.csv", 'Iowa (won) — choppy, 2 locks but realized losses', 58)
plot_game("Washington St. at Oregon St._KXNCAAMBGAME-26FEB04WSUORST-WSU_20260205_040959.csv", 'WSU (lost) — only 1 lock, two realized exits', 60)