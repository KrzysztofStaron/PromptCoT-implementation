# em_logging.py
# Logging helpers for EM training loop
import wandb
import logging

log = logging.getLogger(__name__)

# === WANDB LOGGING HELPERS ===

def log_batch_metrics(em_iter, batch_num, total_batches, batch_rewards, batch_selected_rewards, 
                      avg_rationale_length, batch_tiebreaker_used, batch_reward_spreads_eligible, 
                      batch_eligible_count, batch_size):
    """Log batch-level metrics to wandb"""
    if batch_rewards:
        avg_reward = sum(batch_rewards) / len(batch_rewards)
        max_reward = max(batch_rewards)
        min_reward = min(batch_rewards)
    else:
        avg_reward = max_reward = min_reward = 0.0
    
    # Let wandb auto-increment steps to avoid conflicts when resuming
    wandb.log({
        "batch/iteration": em_iter,
        "batch/batch_num": batch_num,
        "batch/reward_min": min_reward,
        "batch/reward_avg": avg_reward,
        "batch/reward_max": max_reward,
    })
    
    log.info(f"[E-STEP] Batch {batch_num} complete. Reward - Min: {min_reward:.2f}, Avg: {avg_reward:.2f}, Max: {max_reward:.2f}")

def log_final_batch_metrics(em_iter, batch_num, total_batches, batch_rewards, batch_selected_rewards,
                            avg_rationale_length, batch_tiebreaker_used, batch_reward_spreads_eligible,
                            batch_eligible_count, batch_size):
    """Log final batch metrics to wandb"""
    if batch_rewards:
        avg_reward = sum(batch_rewards) / len(batch_rewards)
        max_reward = max(batch_rewards)
        min_reward = min(batch_rewards)
    else:
        avg_reward = max_reward = min_reward = 0.0
    
    # Let wandb auto-increment steps to avoid conflicts when resuming
    wandb.log({
        "batch/iteration": em_iter,
        "batch/batch_num": batch_num,
        "batch/reward_min": min_reward,
        "batch/reward_avg": avg_reward,
        "batch/reward_max": max_reward,
    })
    
    log.info(f"[E-STEP] Final batch complete. Reward - Min: {min_reward:.2f}, Avg: {avg_reward:.2f}, Max: {max_reward:.2f}")

def log_e_step_summary(em_iter, total_batches, all_rewards, total_tiebreaker_used):
    """Log E-step summary to wandb"""
    if all_rewards:
        avg_reward = sum(all_rewards) / len(all_rewards)
        max_reward = max(all_rewards)
        min_reward = min(all_rewards)
        log.info(f"[E-STEP] Complete! Selected {len(all_rewards)} triples")
        log.info(f"[E-STEP] Reward - Min: {min_reward:.2f}, Avg: {avg_reward:.2f}, Max: {max_reward:.2f}")
    else:
        avg_reward = max_reward = min_reward = 0.0
    
    # Let wandb auto-increment steps to avoid conflicts when resuming
    wandb.log({
        "e_step/iteration": em_iter,
        "e_step/reward_min": min_reward,
        "e_step/reward_avg": avg_reward,
        "e_step/reward_max": max_reward,
    })
    log.info(f"[WANDB] Logged E-step summary for iteration {em_iter}: reward_min={min_reward:.2f}, reward_avg={avg_reward:.2f}, reward_max={max_reward:.2f}")

def log_m_step_summary(em_iter, e_step_global_step, prompt_loss, rationale_loss, 
                       prompt_structure_accuracy, rationale_structure_accuracy):
    """Log M-step summary to wandb"""
    losses = [prompt_loss, rationale_loss]
    min_loss = min(losses)
    avg_loss = sum(losses) / len(losses)
    max_loss = max(losses)
    
    # Let wandb auto-increment steps to avoid conflicts when resuming
    wandb.log({
        "m_step/iteration": em_iter,
        "m_step/loss_min": min_loss,
        "m_step/loss_avg": avg_loss,
        "m_step/loss_max": max_loss,
    })
    log.info(f"[WANDB] Logged M-step for iteration {em_iter}: Loss - Min: {min_loss:.4f}, Avg: {avg_loss:.4f}, Max: {max_loss:.4f}")

def log_iteration_summary(em_iter, m_step_global_step, num_triples):
    """Log iteration summary to wandb"""
    # Let wandb auto-increment steps to avoid conflicts when resuming
    wandb.log({
        "iteration/num": em_iter,
    })
    log.info(f"[WANDB] Logged iteration {em_iter} complete: {num_triples} triples")

# === VERBOSE LOGGING HELPERS ===

def log_winner_rationale(best_idx, num_candidates, selected_reward, reward_spread, 
                        problem, rewards, rationale, tiebreaker_used=False):
    """Log details about the selected winning rationale"""
    if tiebreaker_used:
        log.info(f"[WINNER] 🎯 Tiebreaker selected rationale {best_idx+1}/{num_candidates} (reward={selected_reward:.2f}, spread={reward_spread:.3f})")
        log.info(f"[WINNER] Problem: {problem[:100]}..." if len(problem) > 100 else f"[WINNER] Problem: {problem}")
        log.info(f"[WINNER] All candidate rewards: {[f'{r:.3f}' for r in rewards]}")
        log.info(f"[WINNER] Rationale: {rationale[:150]}..." if len(rationale) > 150 else f"[WINNER] Rationale: {rationale}")
    else:
        log.info(f"[WINNER] Selected rationale {best_idx+1}/{num_candidates} (reward={selected_reward:.2f}, spread={reward_spread:.3f})")
        log.info(f"[WINNER] Rationale: {rationale[:150]}..." if len(rationale) > 150 else f"[WINNER] Rationale: {rationale}")

def log_batch_start(batch_num, total_batches, batch_size, current_k_samples):
    """Log batch processing start"""
    log.info(f"[E-STEP] Processing batch {batch_num}/{total_batches} ({batch_size} samples)")
    log.info(f"[E-STEP] Generating {current_k_samples} rationale candidates per sample...")

def log_batch_generation_complete(num_candidates):
    """Log completion of rationale generation"""
    log.info(f"[E-STEP] Generated {num_candidates} total candidates")
    log.info(f"[E-STEP] Computing rewards and selecting best rationale...")

def log_e_step_progress(sample_idx, batch_size):
    """Log progress within E-step batch processing"""
    if (sample_idx + 1) % 4 == 0:
        log.info(f"[E-STEP]   Processed {sample_idx+1}/{batch_size} samples in batch")

