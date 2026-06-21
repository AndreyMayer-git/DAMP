import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import shutil
from matplotlib.lines import Line2D


def classify_conflict(left_emo, right_emo):

    emos = {left_emo, right_emo}
    if emos == {'positive', 'negative'}:
        return 'high'
    else:
        return 'low'


def calculate_actual_deviation(trial_data):

    x, y = trial_data['x'].values, trial_data['y'].values
    if len(x) < 5: return 0

    start = np.array([x[0], y[0]])
    end = np.array([x[-1], y[-1]])
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    if line_len == 0: return 0

    line_unitvec = line_vec / line_len
    distances = []
    for xi, yi in zip(x, y):
        p_vec = np.array([xi, yi]) - start
        dist = np.abs(line_unitvec[0] * p_vec[1] - line_unitvec[1] * p_vec[0])
        distances.append(dist)

    t = trial_data['t_rel'].values
    return np.trapezoid(distances, t)


def run_advanced_analysis(base_data_path="data"):

    session_dirs = [d for d in glob.glob(os.path.join(base_data_path, "*")) if os.path.isdir(d)]

    for session_path in session_dirs:
        session_name = os.path.basename(session_path)
        print(f"\nProcessing session: {session_name}")
        res_p = os.path.join(session_path, 'results.csv')
        traj_p = os.path.join(session_path, 'trajectories.csv')
        if not os.path.exists(res_p) or not os.path.exists(traj_p):
            continue

        df = pd.read_csv(res_p)
        df_t = pd.read_csv(traj_p)

        df_t = df_t[~((df_t['x'] == 0) & (df_t['y'] == 0) & (df_t['t_rel'] > 0.05))]

        base_rt = df['base_rt'].iloc[0]


        anomaly_path = os.path.join(session_path, 'anomaly_trials')
        normal_path = os.path.join(session_path, 'normal_trials')
        for p in [anomaly_path, normal_path]:
            if os.path.exists(p):
                shutil.rmtree(p)
            os.makedirs(p)

        sns.set_theme(style="whitegrid")

        trial_stats = []
        test_df_t = df_t[df_t['type'] != 'CALIBRATION']

        for n in test_df_t['n'].unique():
            t_data = test_df_t[test_df_t['n'] == n].sort_values('t_rel')
            if len(t_data) < 5:
                continue
            auc = calculate_actual_deviation(t_data)

            xs = t_data['x'].values
            committed = np.sign(xs[np.abs(xs) > 20])
            n_crossings = int((np.diff(committed) != 0).sum()) if committed.size > 1 else 0

            info_rows = df[df['n'] == n]
            if info_rows.empty:
                continue
            info = info_rows.iloc[0]

            trial_stats.append({
                'n': n,
                'auc': auc,
                'flip': n_crossings,
                'rt': info['rt'],
                'choice': info['choice'],
                'conflict': classify_conflict(info['emo_l'], info['emo_r']),
                'chosen_emo': info['chosen_emo'],
                'color': info.get('chosen_color', 'N/A')
            })

        df_d = pd.DataFrame(trial_stats)

        emo_df = df[df['chosen_emo'].isin(['positive', 'neutral', 'negative'])]
        if not emo_df.empty:
            emo_counts = emo_df['chosen_emo'].value_counts(normalize=True) * 100
            emo_counts = emo_counts.reindex(['positive', 'neutral', 'negative']).fillna(0)

            plt.figure(figsize=(6, 5))
            ax = sns.barplot(
                x=emo_counts.index,
                y=emo_counts.values,
                hue=emo_counts.index,
                palette={
                    'positive': '#4CAF50',
                    'neutral': '#9E9E9E',
                    'negative': '#F44336'
                },
                legend=False
            )

            plt.ylabel('Choice frequency (%)')
            plt.xlabel('Emotional category')
            plt.title('Emotion Choice Frequency')

            for i, v in enumerate(emo_counts.values):
                ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

            plt.ylim(0, 100)
            plt.savefig(os.path.join(session_path, 'emotion_choice_frequency.png'), dpi=150)
            plt.close()

        auc_high = df_d[df_d['conflict'] == 'high']['auc'].mean()
        auc_low = df_d[df_d['conflict'] == 'low']['auc'].mean()
        DCI = (auc_high - auc_low) / auc_low if auc_low > 0 else np.nan

        sw = df[df['type'] == 'SWAP']
        RSI = (sw['is_swap'] == 0).mean() if not sw.empty else np.nan

        FLIP = df_d['flip'].mean() if not df_d.empty else np.nan

        avg_auc = df_d['auc'].mean() if not df_d.empty else 0

        if 'chosen_color' in df.columns:
            all_appeared_colors = pd.concat([df['color_l'], df['color_r']])
            appearance_counts = all_appeared_colors.value_counts()

            choice_counts = df[~df['chosen_color'].isin(['none', 'error', 'N/A'])]['chosen_color'].value_counts()

            min_appearances = 3
            valid_colors = appearance_counts[appearance_counts >= min_appearances].index

            color_preference_pct = (choice_counts / appearance_counts * 100).fillna(0)
            color_preference_pct = color_preference_pct[color_preference_pct.index.isin(valid_colors)].sort_values(
                ascending=False)

            if not color_preference_pct.empty:
                plt.figure(figsize=(10, 6))
                pal = {c: c.lower() if c.lower() != 'white' else '#f0f0f0' for c in color_preference_pct.index}

                ax = sns.barplot(x=color_preference_pct.index, y=color_preference_pct.values,
                                 hue=color_preference_pct.index, palette=pal, legend=False)

                plt.axhline(50, color='black', linestyle='--', alpha=0.5, label='Random (50%)')
                plt.title(
                    f'Net color preference\n(% of choices from total presentations, min {min_appearances} appearances)')
                plt.ylabel('Preference, %')

                for i, c in enumerate(color_preference_pct.index):
                    v = color_preference_pct[c]
                    n_chosen = choice_counts.get(c, 0)
                    n_appeared = appearance_counts[c]
                    ax.text(i, v + 2, f'{int(v)}%\n({n_chosen}/{n_appeared})',
                            ha='center', fontweight='bold', fontsize=9)

                plt.legend()
                plt.savefig(os.path.join(session_path, 'color_preference.png'), dpi=150)
                plt.close()

            plt.figure(figsize=(10, 6))
            df_rt_color = df[~df['chosen_color'].isin(['none', 'error', 'N/A']) & (df['rt'] > 0)].copy()

            if not df_rt_color.empty:
                avg_rt_color = df_rt_color.groupby('chosen_color')['rt'].mean().sort_values()
                pal_rt = {c: c.lower() if c.lower() != 'white' else '#f0f0f0' for c in avg_rt_color.index}

                ax_rt = sns.barplot(x=avg_rt_color.index, y=avg_rt_color.values,
                                    hue=avg_rt_color.index, palette=pal_rt, legend=False)

                plt.axhline(base_rt, color='red', linestyle='--', alpha=0.6,
                            label=f'Base RT ({int(base_rt)}ms)')

                for i, v in enumerate(avg_rt_color.values):
                    ax_rt.text(i, v + 5, f'{int(v)}ms', ha='center', fontweight='bold')

                plt.title('Mean RT by color')
                plt.ylabel('RT (ms)')
                plt.legend()
                plt.savefig(os.path.join(session_path, 'rt_by_color.png'), dpi=150)
                plt.close()

        if not sw.empty:
            plt.figure(figsize=(6, 6))
            sc = sw['is_swap'].value_counts()
            plt.pie([sc.get(0, 0), sc.get(1, 0)], labels=['Consistent', 'Switched'], autopct='%1.1f%%',
                    colors=['#2196F3', '#FFC107'])
            plt.title('SWAP Test Consistency')
            plt.savefig(os.path.join(session_path, 'swap_consistency.png'))
            plt.close()

        plt.figure(figsize=(12, 5))
        sns.lineplot(data=df[df['rt'] > 0], x='n', y='rt', color='gray', alpha=0.3)
        sns.scatterplot(data=df[df['rt'] > 0], x='n', y='rt', hue='type', s=80, palette='bright')
        plt.axhline(base_rt, color='red', linestyle='--', label=f'Calibration ({int(base_rt)}ms)')
        plt.title('Reaction time (RT)')
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.savefig(os.path.join(session_path, 'rt_dynamics.png'), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 5))
        if 'conflict' in df_d.columns:
            sns.barplot(data=df_d, x='n', y='auc', hue='conflict', palette='viridis', dodge=False)
        else:
            sns.barplot(data=df_d, x='n', y='auc', color='steelblue')
        plt.axhline(avg_auc, color='black', linestyle=':', alpha=0.5, label='Average')
        plt.title('Cognitive effort index (AUC)')
        plt.legend()
        plt.savefig(os.path.join(session_path, 'auc_profile.png'))
        plt.close()

        for _, row in df_d.iterrows():
            is_anomaly = row['flip'] >= 1 or row['auc'] > avg_auc * 1.5
            curr_path = anomaly_path if is_anomaly else normal_path
            t_data = test_df_t[test_df_t['n'] == row['n']].sort_values('t_rel')

            fig, ax = plt.subplots(figsize=(8, 8))
            START_POS = (0, -500)
            click_x, click_y = t_data['x'].iloc[-1], t_data['y'].iloc[-1]

            ax.grid(True, linestyle=':', alpha=0.6)
            ax.axvline(0, color='black', alpha=0.2, lw=1)
            ax.axhline(0, color='black', alpha=0.2, lw=1)
            ax.add_patch(plt.Rectangle((-900, -100), 400, 400, color='gray', alpha=0.05))
            ax.add_patch(plt.Rectangle((500, -100), 400, 400, color='gray', alpha=0.05))

            ax.plot([START_POS[0], click_x], [START_POS[1], click_y], 'g--', alpha=0.3, label='Perfect route',
                    zorder=1)
            ax.plot(START_POS[0], START_POS[1], 'ro', markersize=10, label='Start (0, -500)', zorder=5,
                    markeredgecolor='black')
            sc = ax.scatter(t_data['x'], t_data['y'], c=t_data['t_rel'], cmap='autumn', s=35, zorder=4,
                            edgecolor='white', linewidth=0.3)
            ax.plot(t_data['x'], t_data['y'], 'k-', alpha=0.2, lw=1)
            ax.plot(click_x, click_y, 'go', markersize=10, label='Click (END)', zorder=6, markeredgecolor='black')

            cbar = plt.colorbar(sc)
            cbar.set_label('Time (ms)', rotation=270, labelpad=15)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

            ax.set_title(f"Trial #{row['n']} ({row['color']})\nAUC: {int(row['auc'])} | RT: {int(row['rt'])}ms")
            ax.set_xlim(-960, 960)
            ax.set_ylim(-600, 600)
            plt.savefig(os.path.join(curr_path, f"trial_{row['n']}_trace.png"), bbox_inches='tight')
            plt.close()



        plt.figure(figsize=(8, 5))
        names = ['DCI', 'RSI', 'FLIP']
        vals = [DCI, RSI, FLIP]

        ax = sns.barplot(x=names, y=vals, hue=names, palette='Set2', legend=False)
        plt.axhline(0, color='black', lw=1)
        plt.title(f'Decision Dynamics Profile\nSession: {session_name}')
        plt.ylabel('Index value')

        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(i, v + (0.02 if v >= 0 else -0.02), f'{v:.2f}', ha='center',
                        va='bottom' if v >= 0 else 'top', fontweight='bold')

        plt.savefig(os.path.join(session_path, 'indices.png'), dpi=150, bbox_inches='tight')
        plt.close()




        try:
            age = df['age'].iloc[0] if 'age' in df.columns else "N/A"
            gender = df['gender'].iloc[0] if 'gender' in df.columns else "N/A"


            pd.DataFrame([{
                'session': session_name,
                'age': age,
                'gender': gender,

                'DCI': DCI,
                'RSI': RSI,
                'FLIP': FLIP
            }]).to_csv(os.path.join(session_path, 'indices.csv'), index=False)


            with open(os.path.join(session_path, 'scientific_summary.txt'), 'w', encoding='utf-8') as f:
                f.write("═══════════════════════════════════════════════════════\n")
                f.write("DECISION DYNAMICS PROFILE\n")
                f.write("═══════════════════════════════════════════════════════\n\n")
                f.write(f"Session: {session_name}\n")
                f.write("───────────────────────────────────────────────────────\n")
                f.write("DECISION DYNAMICS INDICES\n")
                f.write("───────────────────────────────────────────────────────\n")
                f.write(f"DCI (Decision Conflict):          {DCI:.3f}\n")
                f.write(f"RSI (Stability):                  {RSI:.3f}\n")
                f.write(f"FLIP (mean centerline crossings): {FLIP:.3f}\n\n")
                f.write("───────────────────────────────────────────────────────\n")
                f.write("NOTE\n")
                f.write("───────────────────────────────────────────────────────\n")
                f.write("Indices reflect individual differences in emotion-modulated\n")
                f.write("decision dynamics under time pressure.\n")
                f.write("No clinical interpretation is assumed.\n")
                f.write("\n═══════════════════════════════════════════════════════\n")

            print(f"✓ Indices successfully calculated for {session_name}")
            print(f"  DCI={DCI:.3f} | RSI={RSI:.3f} | FLIP={FLIP:.3f}")

        except Exception as e:
            print(f"❌ Error calculating indices for {session_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        plt.close('all')


if __name__ == "__main__":
    run_advanced_analysis()