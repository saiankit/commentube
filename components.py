def card(cluster_number):
    return f"""
    <div class="card bg-dark m-4" style="width: 18rem;">
        <div class="card-body">
            <h5 class="card-title">Cluster {cluster_number}</h5>
            <p class="card-text">Cluster Topic Mentioned here</p>
            <a href="#" class="btn btn-primary bg-dark">Summarize Cluster</a>
        </div>
    </div>
    """
