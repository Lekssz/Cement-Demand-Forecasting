# ============================================================
# HELPER FUNCTION: Filter DataFrame by Site
# ============================================================

def get_site_data(df, site_id):
    """
    Filter the DataFrame for a specific site.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The full operations DataFrame
    site_id : str or int
        The site identifier to filter by
    
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame containing only rows for the specified site
    """
    return df[df['site_id'] == site_id].copy()


# Example usage:
# site_1_data = get_site_data(df, 'Site_1')
# site_1_data.head()