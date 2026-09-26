# Personal Planning Sandbox Data Structure

## Sensitivity Levels

- low: non-sensitive metadata, public links, app settings.
- medium: personal routines, private notes, contact info.
- high: financial transactions, health details, private communications.
- critical: credentials, government IDs, bank account identifiers, security recovery data.

## Time Span Policy

- 30d: notifications, short-lived system signals.
- 90d: browser history, app usage, transient activity logs.
- 180d: primary planning context (most inbox and document activity).
- 365d: yearly planning and recurring pattern discovery.
- 24m: finance and relationship trends for better planning decisions.
- 84m: tax/compliance records where realistic retention is needed.
- snapshot: current-state only data (credentials, profile, active settings).

Default planning context should prioritize recent 180d, with targeted retrieval from longer windows.

## env_assets Tree Structure

env_assets/
|- meta/
|  |- user_profile.json                          # snapshot, high
|  |- household_profile.json                     # snapshot, high
|  |- timezone_locale.json                       # snapshot, low
|  |- retention_policy.json                      # snapshot, low
|  |- sensitivity_policy.json                    # snapshot, low
|  |- source_catalog.json                        # snapshot, low
|
|- people/
|  |- contacts/
|  |  |- personal_contacts.json                  # 24m, medium
|  |  |- work_contacts.json                      # 24m, medium
|  |  |- emergency_contacts.json                 # snapshot, high
|  |- relationship_graph/
|  |  |- people_entities.json                    # snapshot + updates, medium
|  |  |- interaction_edges_365d.json             # 365d, medium
|
|- calendar/
|  |- working/
|  |  |- events_past_12m_future_6m.json          # rolling 18m, medium
|  |  |- recurring_rules.json                    # snapshot + updates, medium
|  |  |- meeting_prep_templates.json             # 365d, low
|  |- personal/
|  |  |- events_past_12m_future_12m.json         # rolling 24m, medium
|  |  |- routines_and_habits.json                # 365d, medium
|  |- constraints/
|  |  |- hard_constraints.json                   # snapshot + updates, high
|  |  |- travel_blocks_24m.json                  # 24m, medium
|  |  |- preferred_working_hours.json            # snapshot + updates, medium
|
|- tasks/
|  |- todo_manager/
|  |  |- inbox_tasks_180d.json                   # 180d, medium
|  |  |- active_tasks.json                       # snapshot + updates, medium
|  |  |- completed_tasks_365d.json               # 365d, medium
|  |  |- recurring_tasks.json                    # 24m, medium
|  |- project_plans/
|  |  |- milestones_24m.json                     # 24m, medium
|  |  |- dependencies_24m.json                   # 24m, medium
|
|- email/
|  |- inbox/
|  |  |- threads_180d.json                       # 180d, high
|  |- sent/
|  |  |- threads_180d.json                       # 180d, high
|  |- starred_flagged/
|  |  |- starred_365d.json                       # 365d, high
|  |  |- flagged_action_required_365d.json       # 365d, high
|  |- folders/
|  |  |- receipts_24m.json                       # 24m, high
|  |  |- travel_24m.json                         # 24m, medium
|  |  |- legal_finance_84m.json                  # 84m, critical
|
|- messaging/
|  |- chat_apps/
|  |  |- work_chat_180d.json                     # 180d, high
|  |  |- personal_chat_180d.json                 # 180d, high
|  |  |- pinned_threads_365d.json                # 365d, high
|
|- browser/
|  |- bookmarks/
|  |  |- work_bookmarks.json                     # snapshot + updates, medium
|  |  |- personal_bookmarks.json                 # snapshot + updates, medium
|  |- history/
|  |  |- visits_90d.json                         # 90d, high
|  |  |- search_queries_90d.json                 # 90d, high
|  |  |- downloads_90d.json                      # 90d, medium
|  |- sessions/
|  |  |- open_tabs_snapshot.json                 # snapshot, medium
|  |  |- recently_closed_30d.json                # 30d, medium
|
|- notes/
|  |- quick_notes/
|  |  |- notes_180d.json                         # 180d, medium
|  |- knowledge/
|  |  |- evergreen_notes.json                    # snapshot + updates, medium
|  |  |- meeting_notes_365d.json                 # 365d, medium
|  |- attachments/
|  |  |- note_attachment_index_365d.json         # 365d, medium
|
|- desktop_apps/
|  |- password_manager/
|  |  |- vault_items_current.json                # snapshot, critical
|  |  |- credential_usage_90d.json               # 90d, critical
|  |  |- recovery_material_current.json          # snapshot, critical
|  |- finance_apps/
|  |  |- bank_accounts_snapshot.json             # snapshot, critical
|  |  |- statements_24m.json                     # 24m, critical
|  |- notes_apps/
|  |  |- synced_notebooks_365d.json              # 365d, high
|  |- productivity_apps/
|  |  |- app_task_sync_state.json                # snapshot + updates, low
|  |  |- app_usage_90d.json                      # 90d, medium
|
|- finance/
|  |- banking/
|  |  |- transactions_24m.json                   # 24m, critical
|  |  |- recurring_payments_24m.json             # 24m, critical
|  |  |- account_alerts_365d.json                # 365d, high
|  |- cards/
|  |  |- card_transactions_24m.json              # 24m, critical
|  |  |- card_autopay_rules.json                 # snapshot + updates, critical
|  |- income_tax/
|  |  |- payroll_24m.json                        # 24m, critical
|  |  |- tax_documents_84m.json                  # 84m, critical
|  |- investments/
|  |  |- portfolio_snapshot.json                 # snapshot, critical
|  |  |- transactions_36m.json                   # 36m, critical
|
|- health/
|  |- fitness/
|  |  |- workouts_12m.json                       # 12m, high
|  |  |- sleep_180d.json                         # 180d, high
|  |  |- wearable_daily_180d.json                # 180d, high
|  |- medical/
|  |  |- appointments_24m.json                   # 24m, critical
|  |  |- medications_current.json                # snapshot + updates, critical
|  |  |- insurance_details_current.json          # snapshot, critical
|
|- travel/
|  |- plans/
|  |  |- trips_24m.json                          # 24m, medium
|  |  |- reservations_24m.json                   # 24m, high
|  |- loyalty/
|  |  |- programs_snapshot.json                  # snapshot + updates, medium
|
|- home_life/
|  |- household/
|  |  |- chores_365d.json                        # 365d, low
|  |  |- maintenance_schedule_24m.json           # 24m, medium
|  |- shopping/
|  |  |- grocery_patterns_180d.json              # 180d, medium
|  |  |- subscriptions_24m.json                  # 24m, high
|
|- security/
|  |- identity_docs/
|  |  |- government_ids_current.json             # snapshot, critical
|  |  |- insurance_cards_current.json            # snapshot, critical
|  |- auth_events/
|  |  |- sign_in_alerts_365d.json                # 365d, high
|  |  |- device_trust_list_current.json          # snapshot + updates, high
|
|- files/
|  |- documents/
|  |  |- file_index_180d.json                    # 180d, medium
|  |  |- important_docs_index.json               # snapshot + updates, high
|  |- desktop_downloads/
|  |  |- desktop_snapshot_30d.json               # 30d, medium
|  |  |- downloads_index_90d.json                # 90d, medium
|
|- notifications/
|  |- unified/
|  |  |- pending_notifications_7d.json           # 7d, low
|  |  |- notification_history_30d.json           # 30d, medium
|
|- agent_runtime_seed/
|  |- user_intent_profile.json                   # snapshot + updates, medium
|  |- planning_preferences.json                  # snapshot + updates, medium
|  |- privacy_redaction_rules.json               # snapshot, high
|  |- tool_permissions_seed.json                 # snapshot, high
